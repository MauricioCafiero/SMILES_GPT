"""
GPT model classes — architecture from ~/python_mac/gpt/code/CafChemGPT.py
(TokenAndPositionEmbedding, TransformerBlock, GPT), extended with KV-cache
decoding for generation. cache=None reproduces the reference behavior
exactly; with cache, forward processes new tokens only and each block gets a
dict {'k','v'} of shape (B, h, T_past, hd).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Transformer block with multi-head causal self-attention (SDPA).

    q/k/v project embed_dim -> num_heads*key_dim (matching the reference's
    TF-derived parameterization).
    """

    def __init__(self, num_heads, key_dim, embed_dim, ff_dim,
                 dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        qkv_dim = num_heads * key_dim
        self.q_proj = nn.Linear(embed_dim, qkv_dim)
        self.k_proj = nn.Linear(embed_dim, qkv_dim)
        self.v_proj = nn.Linear(embed_dim, qkv_dim)
        self.out_proj = nn.Linear(qkv_dim, embed_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn_1 = nn.Linear(embed_dim, ff_dim)
        self.ffn_2 = nn.Linear(ff_dim, embed_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, cache=None):
        B, L, _ = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, L, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, h, hd).transpose(1, 2)
        if cache is not None and cache.get("k") is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            if L > 1:
                # multi-token update with a past is not needed for decode;
                # prefill always runs without a cache
                raise ValueError("cached forward supports single-token input only")
            attn = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, L, h * hd)
        attn_out = self.dropout_1(self.out_proj(attn))
        out1 = self.ln_1(x + attn_out)
        ffn = self.ffn_2(F.relu(self.ffn_1(out1)))
        ffn = self.dropout_2(ffn)
        return self.ln_2(out1 + ffn), {"k": k, "v": v}


class TokenAndPositionEmbedding(nn.Module):
    """Embeds tokens and positions (GPT-2-style std=0.02 init)."""

    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x, pos_offset=0):
        ln = x.size(1)
        positions = self.pos_emb(
            torch.arange(pos_offset, pos_offset + ln, device=x.device))
        return self.token_emb(x) + positions


class GPT(nn.Module):
    """A small decoder-only GPT producing token logits over the vocabulary."""

    def __init__(self, num_blocks, max_length, vocab_size,
                 embed_dim=256, num_heads=4, key_dim=None,
                 ff_dim=1024, dropout_rate=0.1):
        super().__init__()
        if key_dim is None:
            key_dim = embed_dim // num_heads
        self.num_blocks = num_blocks
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.embedding = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(num_heads, key_dim, embed_dim, ff_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, cache=None):
        if cache is None:
            cache = [dict(k=None, v=None)] * self.num_blocks
        pos = 0 if cache[0]["k"] is None else cache[0]["k"].size(2)
        h = self.embedding(x, pos_offset=pos)
        new_cache = []
        for block, c in zip(self.blocks, cache):
            h, c2 = block(h, c)
            new_cache.append(c2)
        logits = self.head(h)
        return logits, new_cache

    def config(self):
        return {
            "num_blocks": self.num_blocks,
            "max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        }