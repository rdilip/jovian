"""
Code for multiGPT model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass

from jovian.models.rotary import RotaryEmbedding

from jaxtyping import Int, Array


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert self.dim % self.n_heads == 0
        self.dim_per_head = self.dim // self.n_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        # This doesn't need to be in every single attention layer, but the parameters also aren't trainable,
        # so we only pay a small per GPU memory cost.
        self.pos_embed = RotaryEmbedding(self.dim_per_head)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.to_qkv(x).split(self.dim, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, L, self.n_heads, self.dim_per_head).transpose(1, 2),
            (q, k, v),
        )  # (B, H, L, D)
        q, k = self.pos_embed(q, k)
        # y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        # q, k = self.pos_embed(q), self.pos_embed(k)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = self.attn_dropout(y)
        y = y.transpose(1, 2).contiguous().view(B, L, self.dim)
        y = self.resid_dropout(self.to_out(y))
        return y


def mlp(dim: int, hidden_dim: int, dropout: float = 0.0):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, dilation: int = 2, dropout: float = 0.0):
        super().__init__()
        self.attn = SelfAttention(dim, n_heads, dropout)
        self.mlp = mlp(dim, dilation * dim, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class MultiGPTConfig:
    vocab_size: int = 21
    dim: int = 128
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0
    block_size: int = 1024
    bos: int = None
    eos: int = None
    pad: int = None
    dilation: int = 2  # MLP hidden dimension factor


class MultiGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # flex attention precompute

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                Block(cfg.dim, cfg.n_heads, dropout=cfg.dropout, dilation=cfg.dilation)
                for _ in range(cfg.n_layers)
            ]
        )
        self.proj = nn.Linear(cfg.dim, cfg.vocab_size)
        self.embed.weight = self.proj.weight
        self.ln = nn.LayerNorm(cfg.dim)

        self.direction_embed = nn.Embedding(2, cfg.dim)
        self.direction_embed.weight.detach().zero_()

    def from_config(cfg):
        cfg = GPTConfig(
            vocab_size=cfg["vocab_size"],
            dim=cfg["dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            dropout=cfg["dropout"],
            block_size=cfg["block_size"],
            bos=cfg["bos"],
            eos=cfg["eos"],
            pad=cfg["pad"],
        )
        return GPT(cfg)

    def forward(self, tokens, direction: Int[Array, "b"], tgt=None):
        device = tokens.device
        b, t = tokens.size()
        # assert t <= self.cfg.block_size
        # NOTE: this should really be cached, but I'm not sure how to cache it with the max
        # sequence length then take a subset.

        # in context conditioning
        h_gen = self.direction_embed(direction).unsqueeze(-2)  # B 1 D

        h = self.embed(tokens)

        # remove dropout
        h = torch.cat((h_gen, h), dim=-2)
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)

        loss = None
        h = h[:, 1:, :]
        if tgt is not None:
            logits = self.proj(h)
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                tgt.view(-1),
                ignore_index=self.cfg.pad,
            )
        else:
            logits = self.proj(h[:, [-1], :])
        return logits, loss

    @torch.no_grad()
    def generate_with_themtor_sampling(
        self, idx, temperature=1.0, max_output_size: int = 1024
    ):
        # lauren's sampling idea
        device = self.embed.weight.device
        idx = idx.unsqueeze(0).to(device)
        direction = torch.tensor([0, 1]).to(device)
        gen_left = idx[0, 0] != self.cfg.bos
        gen_right = idx[0, -1] != self.cfg.eos

        with torch.inference_mode():
            while gen_left or gen_right:
                idx_combined = torch.cat((idx, idx.flip(dims=(-1,))), dim=0)
                logits, _ = self(idx_combined, direction=direction)
                logits = logits / temperature

                logits_right, logits_left = logits.unbind(0)

                logits_left = self.nucleus_filter(logits_left, p=0.9)
                logits_right = self.nucleus_filter(logits_right, p=0.9)

                p_left = F.softmax(logits_left, dim=-1)
                p_right = F.softmax(logits_right, dim=-1)

                idx_next_left = torch.multinomial(p_left, num_samples=1)
                idx_next_right = torch.multinomial(p_right, num_samples=1)

                p_direction = torch.tensor(
                    [p_right[0, idx_next_right.item()], p_left[0, idx_next_left.item()]]
                ).to(device)

                if not gen_left:
                    p_direction[1] = 0.0
                if not gen_right:
                    p_direction[0] = 0.0

                p_direction /= p_direction.sum()
                direction_sample = torch.multinomial(p_direction, num_samples=1).item()

                if direction_sample == 1:
                    idx = torch.cat((idx_next_left, idx), dim=-1)
                else:
                    idx = torch.cat((idx, idx_next_right), dim=-1)

                gen_left = idx[0, 0] != self.cfg.bos
                gen_right = idx[0, -1] != self.cfg.eos
                if idx.size(1) > max_output_size:
                    # print("went too far")
                    break

        # peel off
        if idx[..., 0] == self.cfg.bos:
            idx = idx[..., 1:]
        if idx[..., -1] == self.cfg.eos:
            idx = idx[..., :-1]

        return idx

    @torch.no_grad()
    def generate_bidirectional(
        self,
        idx,
        temperature=1.0,
        max_output_size: int = 1024,
        sampling_method="minp",
        threshold: float = 0.9,
    ):
        # no batching right now for reasons.
        device = self.embed.weight.device
        idx = idx.unsqueeze(0).to(device)
        direction = torch.tensor([0, 1]).to(device)
        gen_left = idx[0, 0] != self.cfg.bos
        gen_right = idx[0, -1] != self.cfg.eos

        with torch.inference_mode():
            while gen_left or gen_right:
                idx_combined = torch.cat((idx, idx.flip(dims=(-1,))), dim=0)
                logits, _ = self(idx_combined, direction=direction)
                logits = logits / temperature

                logits_right, logits_left = logits.unbind(0)
                if sampling_method == "nucleus":
                    assert threshold > 0.8
                    logits_left = self.nucleus_filter(logits_left, p=threshold)
                    logits_right = self.nucleus_filter(logits_right, p=threshold)
                    # look at coupled probabilities
                    p_left = F.softmax(logits_left, dim=-1)
                    p_right = F.softmax(logits_right, dim=-1)
                elif sampling_method == "minp":
                    assert threshold < 0.8  # sanity check we didn't get mixed up
                    p_left = self.minp_filter(logits_left, p_base=threshold)
                    p_right = self.minp_filter(logits_right, p_base=threshold)

                idx_next_left = torch.multinomial(p_left, num_samples=1)
                idx_next_right = torch.multinomial(p_right, num_samples=1)

                p_direction = torch.tensor(
                    [p_right[0, idx_next_right.item()], p_left[0, idx_next_left.item()]]
                ).to(device)

                if not gen_left:
                    p_direction[1] = 0.0
                if not gen_right:
                    p_direction[0] = 0.0

                p_direction /= p_direction.sum()
                direction_sample = torch.multinomial(p_direction, num_samples=1).item()

                if direction_sample == 1:
                    idx = torch.cat((idx_next_left, idx), dim=-1)
                else:
                    idx = torch.cat((idx, idx_next_right), dim=-1)

                gen_left = idx[0, 0] != self.cfg.bos
                gen_right = idx[0, -1] != self.cfg.eos
                if idx.size(1) > max_output_size:
                    # print("went too far")
                    break

        # peel off
        if idx[..., 0] == self.cfg.bos:
            idx = idx[..., 1:]
        if idx[..., -1] == self.cfg.eos:
            idx = idx[..., :-1]

        return idx

    @torch.no_grad()
    def minp_filter(self, logits, p_base=0.1):
        probs = logits.softmax(dim=-1)
        max_token = logits.argmax(dim=-1)
        p_scaled = probs.max(dim=-1).values * p_base
        prob_new = torch.where(
            probs > p_scaled.unsqueeze(-1), probs, torch.zeros_like(probs)
        )
        prob_new = prob_new / prob_new.sum(dim=-1, keepdim=True)

        return prob_new

    @torch.no_grad()
    def nucleus_filter(self, logits, p=0.9):
        """Set logits of tokens outside top-p to -inf."""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        # Set filtered logits to -inf
        sorted_logits = logits.gather(-1, sorted_indices)
        sorted_logits[mask] = float("-inf")

        # Restore original order
        filtered_logits = torch.empty_like(logits).scatter(
            -1, sorted_indices, sorted_logits
        )
        return filtered_logits

    @torch.no_grad()
    def generate_unidirectional(
        self,
        max_output_size,
        mode,
        idx=None,
        temperature=1.0,
        sampling_method="nucleus",
        threshold=0.9,
    ):
        # this is best done with batching, but we should parse for EOS afterwards
        assert mode in ["forward", "reverse"]
        device = self.embed.weight.device
        start = torch.tensor([self.cfg.bos], device=device).unsqueeze(0)

        if mode == "forward":
            start_token = self.cfg.bos
            end_token = self.cfg.eos
            direction = torch.tensor([0]).to(device)
        else:
            start_token = self.cfg.eos
            end_token = self.cfg.bos
            direction = torch.tensor([1]).to(device)

        if idx is None:
            bos = torch.tensor([start_token], device=device)
            idx = bos.unsqueeze(-1)
        else:
            idx = idx.unsqueeze(0).to(device)

        while idx.size(-1) < max_output_size:
            logits, _ = self(idx, direction=direction)
            logits = logits / temperature
            logits[..., start_token] = -1.0e8

            if sampling_method == "nucleus":
                assert threshold > 0.8
                logits = self.nucleus_filter(logits, p=threshold)
                # never predict start token
                probs = F.softmax(logits, dim=-1)
            elif sampling_method == "minp":
                assert threshold < 0.8  # sanity check we didn't get mixed up
                probs = self.minp_filter(logits, p_base=threshold)

            idx_next = torch.multinomial(probs[:, 0, :], num_samples=1)
            if idx_next.item() == end_token:
                break

            idx = torch.cat((idx, idx_next), dim=-1)

            if idx.size(-1) >= max_output_size:
                break
                # print("went too far on mode {}".format(mode))

        if idx[..., 0] == start_token:
            idx = idx[..., 1:]

        if mode == "reverse":
            idx = idx.flip(-1)

        return idx

    def num_params(self):
        N = sum(p.numel() for p in self.parameters())
        N -= self.proj.weight.numel()  # parameter sharing
        return N

    def get_flops(self):
        T = self.cfg.block_size
        L = self.cfg.n_layers
        B = 1
        V = self.cfg.vocab_size
        D = self.cfg.dim
        H = self.cfg.n_heads
        gamma = self.blocks[0].mlp[0].weight.size(-2) / D
        prefactor = 6 if self.training else 2
        matmul = prefactor * L * T * D * D * (2 * gamma + 3 + 1)

        # the D below is actually n_heads * dim_per_head
        att = prefactor * 2 * T * T * D * L
        # weight tying means the parameter count ehre is only used once, the flop
        # add on is from the projection out because the initial embedding has 0 flop
        embed = prefactor * B * T * D * V
        return matmul + embed + att
