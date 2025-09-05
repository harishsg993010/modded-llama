import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import time
import copy
import glob
import math
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward()
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Spiking Neural Network Components

class SurrogateGradient(torch.autograd.Function):
    """Surrogate gradient function for spiking neurons"""
    
    @staticmethod
    def forward(ctx, input, threshold=1.0, slope=25.0):
        ctx.save_for_backward(input)
        ctx.slope = slope
        ctx.threshold = threshold
        return (input > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        slope = ctx.slope
        threshold = ctx.threshold
        
        # Fast sigmoid surrogate gradient
        grad_input = grad_output * slope * torch.sigmoid(slope * (input - threshold)) * (1 - torch.sigmoid(slope * (input - threshold)))
        return grad_input, None, None

class SpikingNeuron(nn.Module):
    """Leaky Integrate-and-Fire (LIF) neuron for attention"""
    def __init__(self, beta=0.9, threshold=1.0, slope=25.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        self.surrogate_spike = SurrogateGradient.apply
        
    def forward(self, input_current, mem=None):
        if mem is None:
            mem = torch.zeros_like(input_current)
        
        # Leaky integration
        mem = self.beta * mem + input_current
        # Generate spikes
        spk = self.surrogate_spike(mem, self.threshold, self.slope)
        # Reset membrane potential
        mem = mem - spk * self.threshold
        
        return spk, mem

# -----------------------------------------------------------------------------
# Liquid Time Constant Components

class LiquidTimeConstantCell(nn.Module):
    """Liquid Time Constant (LTC) cell for adaptive dynamics"""
    def __init__(self, input_size, hidden_size, tau_min=0.1, tau_max=10.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Networks for computing time constants and gates
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.input_transform = nn.Linear(input_size, hidden_size)
        self.hidden_transform = nn.Linear(hidden_size, hidden_size)
        self.gate_net = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input_t, hidden_state, dt=1.0):
        # Concatenate input and hidden state
        combined = torch.cat([input_t, hidden_state], dim=-1)
        
        # Compute adaptive time constants
        tau_norm = self.tau_net(combined)
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_norm
        
        # Compute gating and transformations
        gate = torch.sigmoid(self.gate_net(combined))
        input_contrib = torch.tanh(self.input_transform(input_t))
        hidden_contrib = torch.tanh(self.hidden_transform(hidden_state))
        
        # Compute target state
        target_state = gate * input_contrib + (1 - gate) * hidden_contrib
        
        # Differential equation: dhdt = (-h + target) / tau
        dhdt = (-hidden_state + target_state) / (tau + 1e-8)
        new_hidden = hidden_state + dt * dhdt
        new_hidden = torch.tanh(new_hidden)  # Keep bounded
        
        return new_hidden, new_hidden

# -----------------------------------------------------------------------------
# Core Llama Components

def norm(x: Tensor):
    """RMSNorm as used in Llama"""
    return F.rms_norm(x, (x.size(-1),))

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int, base=10000.0):
        super().__init__()
        # Compute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
        # Precompute cos and sin for efficiency
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len):
        # x shape: [batch, seq_len, heads, head_dim]
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        # Apply rotary embedding
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        return (x * cos[None, :, None, :]) + (rotate_half(x) * sin[None, :, None, :])

class SpikingAttention(nn.Module):
    """Spiking Neural Network enhanced Multi-Head Attention"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Spiking neurons for Q, K, V
        self.spike_q = SpikingNeuron(beta=0.9, threshold=1.0, slope=25.0)
        self.spike_k = SpikingNeuron(beta=0.9, threshold=1.0, slope=25.0)
        self.spike_v = SpikingNeuron(beta=0.9, threshold=1.0, slope=25.0)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # Membrane potential memory
        self.mem_q = None
        self.mem_k = None
        self.mem_v = None

    def reset_states(self):
        """Reset spiking neuron states"""
        self.mem_q = None
        self.mem_k = None
        self.mem_v = None

    def forward(self, x: Tensor):
        batch_size, seq_len, dim = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply spiking dynamics to Q, K, V
        if self.mem_q is None or self.mem_q.shape != q.shape:
            self.mem_q = torch.zeros_like(q)
            self.mem_k = torch.zeros_like(k)
            self.mem_v = torch.zeros_like(v)
        
        spike_q, self.mem_q = self.spike_q(q, self.mem_q)
        spike_k, self.mem_k = self.spike_k(k, self.mem_k)
        spike_v, self.mem_v = self.spike_v(v, self.mem_v)
        
        # Apply rotary embeddings to spiking Q and K
        rope_q = self.rotary_emb(spike_q, seq_len)
        rope_k = self.rotary_emb(spike_k, seq_len)
        rope_v = spike_v  # V doesn't get rotary embedding
        
        # Transpose for attention computation
        rope_q = rope_q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        rope_k = rope_k.transpose(1, 2)
        rope_v = rope_v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(rope_q, rope_k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, rope_v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        
        return out

class LTCFeedForward(nn.Module):
    """Liquid Time Constant enhanced Feed-Forward Network"""
    def __init__(self, dim: int, hidden_dim: int = None, ltc_hidden_size: int = 256, ltc_layers: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        
        # Standard SwiGLU components
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
        # LTC components
        self.ltc_cells = nn.ModuleList([
            LiquidTimeConstantCell(
                hidden_dim if i == 0 else ltc_hidden_size,
                ltc_hidden_size
            ) for i in range(ltc_layers)
        ])
        
        self.ltc_proj = nn.Linear(ltc_hidden_size, hidden_dim, bias=False)
        self.ltc_states = None

    def reset_states(self):
        """Reset LTC states"""
        self.ltc_states = None

    def forward(self, x: Tensor):
        # Standard SwiGLU computation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Initialize LTC states if needed
        if self.ltc_states is None or len(self.ltc_states) != len(self.ltc_cells):
            self.ltc_states = []
            for cell in self.ltc_cells:
                state = torch.zeros(batch_size, seq_len, cell.hidden_size, 
                                  device=x.device, dtype=x.dtype)
                self.ltc_states.append(state)
        
        # Check shape compatibility and reinitialize if needed
        if self.ltc_states[0].shape[:2] != (batch_size, seq_len):
            self.ltc_states = []
            for cell in self.ltc_cells:
                state = torch.zeros(batch_size, seq_len, cell.hidden_size, 
                                  device=x.device, dtype=x.dtype)
                self.ltc_states.append(state)
        
        # Pass through LTC cells
        ltc_input = intermediate
        for i, ltc_cell in enumerate(self.ltc_cells):
            ltc_output, self.ltc_states[i] = ltc_cell(ltc_input, self.ltc_states[i])
            ltc_input = ltc_output
        
        # Enhance intermediate representation with LTC output
        enhanced = intermediate + self.ltc_proj(ltc_output)
        
        # Final projection
        output = self.down_proj(enhanced)
        return output

class TransformerBlock(nn.Module):
    """Llama Transformer Block with Spiking Attention and LTC Feed-Forward"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, hidden_dim: int = None):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = SpikingAttention(dim, num_heads, max_seq_len)
        self.ffn_norm = nn.RMSNorm(dim)
        self.feed_forward = LTCFeedForward(dim, hidden_dim)

    def reset_states(self):
        """Reset all dynamic states"""
        self.attention.reset_states()
        self.feed_forward.reset_states()

    def forward(self, x: Tensor):
        # Attention with residual connection
        h = x + self.attention(self.attention_norm(x))
        # Feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# -----------------------------------------------------------------------------
# Main Arthemis Model

class ArthemisLlama1B(nn.Module):
    """1B parameter Llama model with Spiking Attention and LTC Feed-Forward"""
    def __init__(
        self, 
        vocab_size: int = 50257,
        dim: int = 2048,
        num_layers: int = 22,
        num_heads: int = 16,
        max_seq_len: int = 2048,
        hidden_dim: int = None
    ):
        super().__init__()
        
        # Model configuration
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = dim // num_heads
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, max_seq_len, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Arthemis Model initialized with {total_params/1e9:.2f}B parameters")

    def _init_weights(self, module):
        """Initialize weights following Llama conventions"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_states(self):
        """Reset all spiking and LTC states"""
        for block in self.blocks:
            block.reset_states()

    def forward(self, input_ids: Tensor, targets: Tensor = None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        if targets is not None:
            # Shift targets for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            return loss
        
        return logits

# -----------------------------------------------------------------------------
# Distributed Training Components

class DistributedAdam(torch.optim.Optimizer):
    """Distributed Adam optimizer"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

# -----------------------------------------------------------------------------
# Data Loading

def load_data_shard(filename: str):
    """Load data shard"""
    with open(filename, 'rb') as f:
        # Skip header (assuming standard format)
        f.seek(1024)  
        data = torch.frombuffer(f.read(), dtype=torch.uint16)
    return data.long()

def create_data_loader(data_pattern: str, seq_len: int, batch_size: int):
    """Create distributed data loader"""
    files = sorted(glob.glob(data_pattern))
    
    def data_generator():
        for file in files:
            data = load_data_shard(file)
            for i in range(0, len(data) - seq_len, batch_size * seq_len):
                batch_data = data[i:i + batch_size * seq_len + 1]
                if len(batch_data) < batch_size * seq_len + 1:
                    continue
                
                inputs = batch_data[:-1].view(batch_size, seq_len)
                targets = batch_data[1:].view(batch_size, seq_len)
                
                yield inputs.cuda(), targets.cuda()
    
    return data_generator()

# -----------------------------------------------------------------------------
# Training Configuration

@dataclass
class TrainingConfig:
    # Model - GPT-2 tokenizer compatible
    vocab_size: int = 50257  # GPT-2 tokenizer: 50,256 tokens + 1 pad token
    dim: int = 2048
    num_layers: int = 22
    num_heads: int = 16
    max_seq_len: int = 2048
    
    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 10000
    warmup_steps: int = 1000
    
    # Data - assuming GPT-2 tokenized data format
    train_data: str = "data/train_*.bin"
    val_data: str = "data/val_*.bin"
    
    # Logging
    log_interval: int = 100
    val_interval: int = 500
    save_interval: int = 1000
    
    # GPT-2 specific tokens
    pad_token_id: int = 50256  # GPT-2 uses last token as pad
    eos_token_id: int = 50256  # End of sequence token

def get_lr_schedule(step: int, config: TrainingConfig):
    """Learning rate schedule with warmup and cosine decay"""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    else:
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

# -----------------------------------------------------------------------------
# Main Training Loop

def train_arthemis():
    """Main training function"""
    
    # Initialize distributed training
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = TrainingConfig()
    
    # Initialize model
    model = ArthemisLlama1B4(
        vocab_size=config.vocab_size,
        dim=config.dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    # Wrap model for distributed training
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Initialize optimizer
    optimizer = DistributedAdam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Compile model for faster training
    model = torch.compile(model)
    
    # Create data loaders
    train_loader = create_data_loader(config.train_data, config.max_seq_len, config.batch_size)
    
    # Training loop
    model.train()
    step = 0
    
    if rank == 0:
        print("Starting training...")
    
    for inputs, targets in train_loader:
        if step >= config.max_steps:
            break
        
        # Update learning rate
        lr = get_lr_schedule(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        loss = model(inputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Reset states periodically for fresh dynamics
        if step % 100 == 0:
            if hasattr(model, 'module'):
                model.module.reset_states()
            else:
                model.reset_states()
        
        # Logging
        if rank == 0 and step % config.log_interval == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, LR = {lr:.6f}")
        
        # Validation
        if step % config.val_interval == 0:
            # Validation logic here
            pass
        
        # Save checkpoint with dual optimizers
        if rank == 0 and step % config.save_interval == 0:
            checkpoint = {
                'model': model.state_dict() if world_size == 1 else model.module.state_dict(),
                'optimizer1': optimizer1.state_dict(),  # DistAdam
                'optimizer2': optimizer2.state_dict(),  # Muon
                'step': step,
                'config': config
            }
            torch.save(checkpoint, f'arthemis_checkpoint_{step}.pt')
        
        step += 1
    
    if rank == 0:
        print("Training completed!")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    train_arthemis()
