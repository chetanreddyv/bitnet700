import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from bitnet_model import BitNet, BitNetConfig
import math
import os
from tqdm import tqdm
from datasets import load_dataset
import json
import logging
import sys
from typing import Dict, Any
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def create_optimizer(model, learning_rate=2e-4, weight_decay=0.1, beta1=0.9, beta2=0.95):
    """Create AdamW optimizer with specified parameters."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8
    )
    return optimizer

def get_lr_scheduler(optimizer, num_warmup_steps=375, num_training_steps=100000):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def count_tokens(dataset, tokenizer):
    """Count the total number of tokens in the dataset."""
    total_tokens = 0
    for example in dataset:
        # Count non-padding tokens
        input_ids = example['input_ids']
        total_tokens += sum(1 for x in input_ids if x != tokenizer.pad_token_id)
    return total_tokens

def prepare_redpajama_dataset(tokenizer, max_length=2048):
    """Prepare RedPajama V2 dataset for training, filtering for high-quality English content."""
    # Load the dataset
    dataset = load_dataset("togethercomputer/RedPajama-Data-V2", "default")
    
    def filter_high_quality(example):
        # Parse metadata and quality signals
        meta = json.loads(example['meta'])
        quality_signals = json.loads(example['quality_signals'])
        
        # Filter for English content
        if meta['language'] != 'en':
            return False
            
        # Filter out duplicates
        if quality_signals.get('is_duplicate', False):
            return False
            
        # Filter based on quality signals
        # You can adjust these thresholds based on your needs
        quality_score = quality_signals.get('quality_score', 0)
        if quality_score < 0.7:  # Only keep high-quality content
            return False
            
        # Filter based on content length
        content_length = len(example['raw_content'].split())
        if content_length < 100 or content_length > 10000:  # Filter out very short or very long content
            return False
            
        return True
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["raw_content"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal language modeling)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Filter for high-quality English content
    dataset = dataset.filter(filter_high_quality)
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Count tokens
    total_tokens = count_tokens(tokenized_dataset, tokenizer)
    print(f"Total number of tokens in filtered dataset: {total_tokens:,}")
    print(f"Approximate number of training examples: {len(tokenized_dataset):,}")
    
    return tokenized_dataset

def validate_model_config(config: BitNetConfig) -> bool:
    """Validate model configuration."""
    try:
        # Check model size
        expected_params = 4 * config.hidden_size * config.num_hidden_layers * (config.hidden_size + 2 * config.intermediate_size)
        if not (680_000_000 <= expected_params <= 720_000_000):
            logger.error(f"Model size {expected_params:,} parameters is not close to 700M")
            return False
            
        # Check head dimension
        if config.hidden_size % config.num_attention_heads != 0:
            logger.error("Hidden size must be divisible by number of attention heads")
            return False
            
        # Check latent dimension
        if config.latent_dim > config.hidden_size // config.num_attention_heads:
            logger.error("Latent dimension cannot be larger than head dimension")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating model config: {str(e)}")
        return False

def validate_dataset(dataset, tokenizer) -> bool:
    """Validate dataset integrity."""
    try:
        # Check dataset size
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return False
            
        # Check token distribution
        token_counts = []
        for example in dataset:
            input_ids = example['input_ids']
            token_count = sum(1 for x in input_ids if x != tokenizer.pad_token_id)
            token_counts.append(token_count)
            
        avg_tokens = sum(token_counts) / len(token_counts)
        if avg_tokens < 100 or avg_tokens > 10000:
            logger.error(f"Average token count {avg_tokens:.2f} is outside expected range")
            return False
            
        # Check for NaN or inf values
        for example in dataset:
            if torch.isnan(torch.tensor(example['input_ids'])).any() or torch.isinf(torch.tensor(example['input_ids'])).any():
                logger.error("Dataset contains NaN or inf values")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return False

def validate_training_setup(model, optimizer, scheduler, device) -> bool:
    """Validate training setup."""
    try:
        # Check model parameters
        if not all(p.requires_grad for p in model.parameters()):
            logger.error("Some model parameters are not set to require gradients")
            return False
            
        # Check optimizer
        if not isinstance(optimizer, optim.AdamW):
            logger.error("Optimizer is not AdamW")
            return False
            
        # Check device
        if not next(model.parameters()).is_cuda:
            logger.error("Model is not on GPU")
            return False
            
        # Check memory usage
        if torch.cuda.get_device_properties(0).total_memory < 16 * 1024 * 1024 * 1024:  # 16GB
            logger.error("GPU memory is less than 16GB")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating training setup: {str(e)}")
        return False

def save_checkpoint(model, optimizer, scheduler, step, save_dir: str):
    """Save training checkpoint with error handling."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"checkpoint-{step}.pt")
        
        # Save to temporary file first
        temp_path = checkpoint_path + ".tmp"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "total_steps": step,
        }, temp_path)
        
        # Move to final location
        os.rename(temp_path, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    save_dir="checkpoints"
):
    """Training loop for BitNet model with error handling."""
    model.train()
    total_steps = 0
    best_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass
                    outputs = model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.error("NaN loss detected")
                        raise ValueError("NaN loss")
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        total_steps += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            "loss": loss.item() * gradient_accumulation_steps,
                            "lr": scheduler.get_last_lr()[0]
                        })
                        
                        # Save checkpoint if loss improved
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            save_checkpoint(model, optimizer, scheduler, total_steps, save_dir)
                    
                    # Clear memory
                    del outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error in training step {step}: {str(e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
        # Save emergency checkpoint
        save_checkpoint(model, optimizer, scheduler, total_steps, save_dir)
        raise

def main():
    try:
        # Calculate model size to achieve 700M parameters
        hidden_size = 2048
        intermediate_size = 5632
        num_hidden_layers = 24
        
        # Model configuration
        config = BitNetConfig(
            vocab_size=32000,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=32,
            intermediate_size=intermediate_size,
            max_position_embeddings=2048,
            latent_dim=hidden_size // 64,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            activation_function="relu",
            layer_norm_eps=1e-6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            use_cache=True,
            gradient_checkpointing=False
        )
        
        # Validate model configuration
        if not validate_model_config(config):
            raise ValueError("Model configuration validation failed")
        
        # Initialize model and tokenizer
        model = BitNet(config)
        
        # Set up Hugging Face token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
        
        # Initialize LLaMA 3.1 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            use_auth_token=True
        )
        
        # Set tokenizer special tokens
        tokenizer.pad_token_id = config.pad_token_id
        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.eos_token_id = config.eos_token_id
        
        # Prepare RedPajama dataset
        train_dataset = prepare_redpajama_dataset(tokenizer)
        
        # Validate dataset
        if not validate_dataset(train_dataset, tokenizer):
            raise ValueError("Dataset validation failed")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(
            model,
            learning_rate=2e-4,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95
        )
        
        scheduler = get_lr_scheduler(
            optimizer,
            num_warmup_steps=375,
            num_training_steps=100000
        )
        
        # Validate training setup
        if not validate_training_setup(model, optimizer, scheduler, device):
            raise ValueError("Training setup validation failed")
        
        # Train the model
        train(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            num_epochs=1,
            device=device,
            gradient_accumulation_steps=4
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 