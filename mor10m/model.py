"""Small token-choice Mixture-of-Recursions; paper arXiv:2507.10524.

Independent compact implementation, not the authors' trained model. Uses
Middle-Cycle sharing, top-1 token-choice, recursion-wise sparse attention,
and the reference code's weighted output skip connection. No future top-k.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
import flashskip_100m as base


@dataclass
class Config(base.FlashSkipConfig):
    d_model: int = 160
    d_ff: int = 496
    n_layers: int = 7
    n_q_heads: int = 5
    n_kv_heads: int = 1
    attention_layers: tuple = tuple(range(7))
    max_seq_len: int = 512
    recursions: int = 3
    balance_coefficient: float = 0.01
    router_z_coefficient: float = 0.001
    model_name: str = "MoR-10M-TokenChoice"
    architecture_version: str = "mor-token-choice-middle-cycle-v1"


def rotary(x, cos, sin):
    # Original token positions retained after sparse packing, not re-numbered.
    c, s = cos[:, None], sin[:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*c-odd*s, even*s+odd*c), -1).flatten(-2)


class Attention(base.GQAAttention):
    def forward(self, x, cos, sin, positions):
        b, t, _ = x.shape
        hq, hk, hd = self.cfg.n_q_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        q = rotary(self.q_norm(self.q(x).view(b,t,hq,hd).transpose(1,2)), cos, sin)
        k = rotary(self.k_norm(self.k(x).view(b,t,hk,hd).transpose(1,2)), cos, sin)
        v = self.v(x).view(b,t,hk,hd).transpose(1,2)
        k, v = k.repeat_interleave(hq//hk,1), v.repeat_interleave(hq//hk,1)
        valid = positions >= 0
        mask = (positions[:,None,:] <= positions[:,:,None]) & valid[:,None,:]
        # Padded queries are discarded; permitting them to see real keys avoids NaNs.
        mask = mask | (~valid[:,:,None] & valid[:,None,:])
        z = F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None],dropout_p=0.)
        return self.o(z.transpose(1,2).reshape(b,t,hq*hd))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = base.RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = base.RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = base.SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self,x,cos,sin,pos):
        x = x + self.attn(self.attn_norm(x),cos,sin,pos)
        return x + self.ffn(self.ffn_norm(x))


class MoR(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        self.tok_embeddings = nn.Embedding(c.vocab_size,c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.router_norm = base.RMSNorm(c.d_model,c.rms_eps)
        self.router = nn.Linear(c.d_model,c.recursions,bias=False)
        self.final_norm = base.RMSNorm(c.d_model,c.rms_eps)
        self.apply(base.FlashSkipLM._init_weights)
        effective = 2+(c.n_layers-2)*c.recursions
        for block in self.blocks:
            nn.init.normal_(block.attn.o.weight,std=.02/math.sqrt(2*effective))
            nn.init.normal_(block.ffn.down.weight,std=.02/math.sqrt(2*effective))

    def forward(self, ids, return_aux=False, forced_depth=None):
        b,t = ids.shape
        if t > self.cfg.max_seq_len: raise ValueError("context exceeds configured limit")
        pos = torch.arange(t,device=ids.device).expand(b,-1)
        cos,sin = base.rope_cos_sin(t,self.cfg.head_dim,self.cfg.rope_theta,ids.device,torch.float32)
        x = self.tok_embeddings(ids)
        x = self.blocks[0](x,cos[pos],sin[pos],pos)
        logits = self.router(self.router_norm(x)).float()
        probs = logits.softmax(-1)
        depth = probs.argmax(-1)
        if forced_depth is not None: depth = forced_depth.expand(b,t)
        freq = F.one_hot(depth,self.cfg.recursions).float().mean((0,1))
        balance = self.cfg.recursions * (freq.detach()*probs.mean((0,1))).sum()
        aux = self.cfg.balance_coefficient*balance + self.cfg.router_z_coefficient*logits.logsumexp(-1).square().mean()
        weight = probs.gather(-1,depth.unsqueeze(-1)).to(x.dtype)
        result, updates = x, x
        for r in range(self.cfg.recursions):
            active = depth >= r
            lengths = active.sum(-1)
            n = int(lengths.max().item())
            if n == 0: break
            # Sorted original indices. Padding is always after real active tokens.
            indices = torch.where(active,pos,t).sort(-1).values[:,:n]
            valid = indices < t
            safe = indices.clamp(max=t-1)
            gather = safe.unsqueeze(-1).expand(-1,-1,x.size(-1))
            z = updates.gather(1,gather)
            packed_pos = indices.masked_fill(~valid,-1)
            for block in self.blocks[1:-1]:
                z = block(z,cos[safe],sin[safe],packed_pos)
            finished = valid & (depth.gather(1,safe)==r)
            result = result.scatter_add(1,gather,z*weight.gather(1,safe.unsqueeze(-1))*finished.unsqueeze(-1))
            # Only active rows enter the next recursion. Padded entries add zero.
            updates = torch.zeros_like(updates).scatter_add(1,gather,z*valid.unsqueeze(-1))
        result = self.blocks[-1](result,cos[pos],sin[pos],pos)
        output = F.linear(self.final_norm(result),self.tok_embeddings.weight)
        if return_aux:
            return output, aux, {"depth_fractions":freq.detach(),"balance_loss":balance.detach()}
        return output


def parameter_count(cfg=None):
    return sum(p.numel() for p in MoR(cfg).parameters())
