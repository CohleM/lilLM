from dataclasses import dataclass 

@dataclass
class Config:
    vocab_size: int = 2**13
    d_model: int = 512
    n_layers: int = 12
    max_seq_len: int = 512
    q_heads: int = 16
    kv_heads: int = 8
    dropout: float = 0.0
    max_batch_size: int = 32
    hidden_dim: int = None
    multiple_of: int = 128
    eps: float = 1e-6
    flash: bool = True
