import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flash_attn import flash_attn_func

def activation_quant(x):
    """Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
        x: an activation tensor with shape [n, d]
    Returns:
        y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    """Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
        w: a weight tensor with shape [d, k]
    Returns:
        u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class SubLN(nn.Module):
    """SubLN normalization as specified in BitNet paper."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = None  # No bias as per BitNet spec

    def forward(self, x):
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale
        return x_norm * self.weight

class BitLinear(nn.Linear):
    """BitLinear layer for BitNet architecture.
    This is only for training, and kernel optimization is needed for efficiency.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.norm = SubLN(in_features)  # Using SubLN instead of RMSNorm

    def forward(self, x):
        w = self.weight
        x_norm = self.norm(x)
        
        # A trick for implementing Straight-Through-Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        y = F.linear(x_quant, w_quant)
        return y

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class BitNetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout if hasattr(config, 'attention_dropout') else 0.0
        self.latent_dim = config.latent_dim if hasattr(config, 'latent_dim') else self.head_dim // 2

        # Projections for latent attention
        self.q_proj = BitLinear(self.hidden_size, self.hidden_size)
        self.k_proj = BitLinear(self.hidden_size, self.hidden_size)
        self.v_proj = BitLinear(self.hidden_size, self.hidden_size)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size)
        
        # Latent space projections (using BitLinear for consistency)
        self.latent_q = BitLinear(self.head_dim, self.latent_dim)
        self.latent_k = BitLinear(self.head_dim, self.latent_dim)
        
        # Initialize RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Project to query, key, value
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Project to latent space and reshape in one step
        q_latent = self.latent_q(q).transpose(1, 2)  # [batch, heads, seq, latent_dim]
        k_latent = self.latent_k(k).transpose(1, 2)  # [batch, heads, seq, latent_dim]
        v = v.transpose(1, 2)  # [batch, heads, seq, head_dim]

        # Compute attention scores in latent space
        attn_weights = torch.matmul(q_latent, k_latent.transpose(-2, -1)) / math.sqrt(self.latent_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, hidden_size)
        
        return self.o_proj(attn_output)

class BitNetFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size)
        self.up_proj = BitLinear(self.hidden_size, self.intermediate_size)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Squared ReLU activation as per BitNet spec
        activation = F.relu(gate) ** 2
        return self.down_proj(activation * up)

class BitNetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = BitNetAttention(config)
        self.ffn = BitNetFFN(config)

    def forward(self, hidden_states, attention_mask=None):
        # Attention
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + attention_output
        
        # FFN
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states

class BitNetConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        intermediate_size=5632,
        max_position_embeddings=2048,
        initializer_range=0.02,
        latent_dim=None,  # If None, will be set to head_dim // 2
        attention_dropout=0.1,
        hidden_dropout=0.1,
        activation_function="relu",
        layer_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        use_cache=True,
        gradient_checkpointing=False,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.latent_dim = latent_dim
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing

class BitNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BitNetBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = SubLN(config.hidden_size)  # Using SubLN instead of RMSNorm
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits 