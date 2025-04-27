# BitNet 700M Implementation

This repository contains an implementation of the BitNet architecture, a 700M parameter language model that uses 1.58-bit quantization for weights and 8-bit quantization for activations.

## Architecture

- Sequence length: 2048
- Transformer-based with BitLinear layers
- Rotary Position Embeddings (RoPE)
- Squared ReLU (ReLU²) activation in FFN layers
- SubLN normalization
- No bias terms in linear or normalization layers
- Multi-head latent attention inspired by deepseek

## Hyperparameters

- Model size: 700M parameters
- Learning rate: 2.0 × 10⁻⁴
- Weight decay: 0.1
- Warmup steps: 375
- Adam beta: (0.9, 0.95)

## Quantization

- Native 1.58-bit weights and 8-bit activations (W1.58A8)
- Weights are quantized to ternary values {-1, 0, +1} using absmean quantization
- Activations are quantized to 8-bit integers using absmax quantization (per-token)
- Model is trained from scratch with this quantization scheme

## Tokenizer

Uses the LLaMA 3 Tokenizer

## Installation

1. Clone the repository:
```bash
[https://github.com/chetanreddyv/bitnet700.git]
cd bitnet700
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

1. Prepare your training data in the appropriate format
2. Modify the `train.py` script to use your dataset
3. Run the training script:
```bash
python train.py
```

### Inference

For inference, you can use the model as follows:

```python
from bitnet_model import BitNet, BitNetConfig
from transformers import AutoTokenizer

# Initialize model and tokenizer
config = BitNetConfig()
model = BitNet(config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load your trained weights
model.load_state_dict(torch.load("path_to_your_weights.pt"))

# Generate text
input_text = "Your input text here"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## Implementation Details

### BitLinear Layer

The BitLinear layer is the core component of the BitNet architecture. It implements:
- Weight quantization to 1.58 bits
- Activation quantization to 8 bits
- Built-in RMSNorm
- Straight-Through Estimator (STE) for training

### Training Process

The model is trained using:
- AdamW optimizer with specified hyperparameters
- Learning rate scheduler with warmup
- Gradient accumulation for effective batch size
- Gradient clipping for stability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the BitNet paper: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2310.11453)
- Uses the LLaMA tokenizer from Meta AI 
