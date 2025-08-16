from dataclasses import dataclass
import socket

VQVAE_CODEBOOK_SIZE = 4096
ESM_SPECIAL_TOKENS = {
    "MASK": VQVAE_CODEBOOK_SIZE,
    "EOS": VQVAE_CODEBOOK_SIZE + 1,
    "BOS": VQVAE_CODEBOOK_SIZE + 2,
    "PAD": VQVAE_CODEBOOK_SIZE + 3,
    "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
}

@dataclass
class TrainConfig:
    batch_size: int = 32
    n_layers: int = 5
    n_heads: int = 8
    dim: int = 64
    dilation: int = 2
    dropout: float = 0.1

    vocab_size: int = VQVAE_CODEBOOK_SIZE + 3
    MASK: int = ESM_SPECIAL_TOKENS["MASK"]
    PAD: int  = ESM_SPECIAL_TOKENS["PAD"]
    BOS: int  = ESM_SPECIAL_TOKENS["BOS"]
    EOS: int  = ESM_SPECIAL_TOKENS["EOS"]

    pth_to_tokens: str = "datasets/ops_swissprot/"
    tokenizer_tag: str = "esm"

    lr: float = 3e-4
    warmup_iters: int = 2_000
    lr_decay_iters: int = 500_000
    min_lr: float = lr / 10
    grad_clip: float = 1.7

    device: str = "cuda:0"
    eval_every: int = 100
    n_eval_batches: int = 10
    save_every: int = 20 * eval_every
    use_wandb: bool = False
    descr: str = ""
    n_samples: int = 8
    attn_type: str = "causal"

    host: str = socket.gethostname()