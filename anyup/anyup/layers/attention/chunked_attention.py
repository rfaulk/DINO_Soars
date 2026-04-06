import torch
import torch.nn as nn
from torch import einsum
from typing import Optional
from .attention_masking import compute_attention_mask


class CrossAttention(nn.Module):
    def __init__(self, qk_dim, num_heads,
                 q_chunk_size: Optional[int] = None,
                 store_attn: bool = False):
        super().__init__()
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.q_chunk_size = q_chunk_size
        self.store_attn = store_attn
        self.attention = nn.MultiheadAttention(
            embed_dim=qk_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    @torch.no_grad()
    def _slice_mask(self, mask, start, end):
        if mask is None:
            return None
        # 2D: (tgt_len, src_len), 3D: (B*num_heads or B, tgt_len, src_len)
        if mask.dim() == 2:
            return mask[start:end, :]
        elif mask.dim() == 3:
            return mask[:, start:end, :]
        else:
            raise ValueError("attn_mask must be 2D or 3D")

    def forward(self, query, key, value, mask=None,
                q_chunk_size: Optional[int] = None,
                store_attn: Optional[bool] = None):
        q_chunk_size = self.q_chunk_size if q_chunk_size is None else q_chunk_size
        store_attn = self.store_attn if store_attn is None else store_attn

        val = key

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Fast path: no chunking
        if q_chunk_size is None or query.size(1) <= q_chunk_size:
            _, attn = self.attention(query, key, val,
                                     average_attn_weights=True,
                                     attn_mask=mask)
            features = einsum("b i j, b j d -> b i d", attn, value)
            return features, (attn if store_attn else None)

        # Chunked over the query length (tgt_len)
        B, Q, _ = query.shape
        outputs = []
        attns = [] if store_attn else None

        for start in range(0, Q, q_chunk_size):
            end = min(start + q_chunk_size, Q)
            q_chunk = query[:, start:end, :]
            mask_chunk = self._slice_mask(mask, start, end)

            # We ignore the MHA output as in JAFAR:
            # use the averaged attention to weight the unprojected V.
            _, attn_chunk = self.attention(q_chunk, key, val,
                                           average_attn_weights=True,
                                           attn_mask=mask_chunk)
            out_chunk = einsum("b i j, b j d -> b i d", attn_chunk, value)
            outputs.append(out_chunk)
            if store_attn:
                attns.append(attn_chunk)

        features = torch.cat(outputs, dim=1)
        attn_scores = torch.cat(attns, dim=1) if store_attn else None
        return features, attn_scores


class CrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1,
                 q_chunk_size: Optional[int] = None, **kwargs):
        super().__init__()
        self.cross_attn = CrossAttention(
            qk_dim, num_heads,
            q_chunk_size=q_chunk_size
        )
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, q_chunk_size: Optional[int] = None, store_attn: Optional[bool] = None, vis_attn=False,
                **kwargs):
        store_attn = store_attn or vis_attn
        q = self.conv2d(q)
        if self.window_ratio > 0:
            attn_mask = compute_attention_mask(
                *q.shape[-2:], *k.shape[-2:], window_size_ratio=self.window_ratio
            ).to(q.device)
        else:
            attn_mask = None
        b, _, h, w = q.shape
        _, _, h_k, w_k = k.shape
        c = v.shape[1]
        q = q.permute(0, 2, 3, 1).view(b, h * w, -1)
        k = k.permute(0, 2, 3, 1).view(b, h_k * w_k, -1)
        v = v.permute(0, 2, 3, 1).view(b, h_k * w_k, -1)

        features, attn = self.cross_attn(q, k, v, mask=attn_mask,
                                         q_chunk_size=q_chunk_size,
                                         store_attn=store_attn)
        features = features.view(b, h, w, c).permute(0, 3, 1, 2)
        if vis_attn:
            from anyup.utils.visualization import visualize_attention_oklab
            import matplotlib.pyplot as plt

            ref, out = visualize_attention_oklab(attn[0], h, w, h_k, w_k)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(ref.cpu().numpy())
            ax[0].set_title("Reference (Values)")
            ax[0].set_xticks([-.5, w_k - .5], labels=[0, w_k])
            ax[0].set_yticks([-.5, h_k - .5], labels=[0, h_k])

            ax[1].imshow(out.cpu().numpy())
            ax[1].set_title("Attention Output")
            ax[1].set_xticks([-.5, w - .5], labels=[0, w])
            ax[1].set_yticks([-.5, h - .5], labels=[0, h])
            plt.show()

        return features
