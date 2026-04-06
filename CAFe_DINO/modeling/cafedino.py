import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from modeling.swin import SwinTransformerBlock

from einops import rearrange, repeat

# PROMPT_TEMPLATES = (
#     "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
#     "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
#     "a close-up photo of {}", "a cropped image featuring {}")

PROMPT_TEMPLATES = ["a photo of a {}"]

class SpatialAggregator(nn.Module):
    def __init__(self, hidden_dim, input_resolution, guidance_dim, heads=8, window_size=7):
        super().__init__()
        self.swin1 = SwinTransformerBlock(dim=hidden_dim, guidance_dim=guidance_dim, input_resolution=input_resolution, num_heads=heads, window_size=window_size, shift_size=0, mlp_ratio=4., qkv_bias=True, drop_path=0.1)
        self.swin2 = SwinTransformerBlock(dim=hidden_dim, guidance_dim=guidance_dim, input_resolution=input_resolution, num_heads=heads, window_size=window_size, shift_size=window_size // 2, mlp_ratio=4., qkv_bias=True, drop_path=0.1)
        self.guidance_norm = nn.LayerNorm(guidance_dim) if guidance_dim > 0 else None

    def forward(self, x, guidance):
        B, D, C, H, W = x.shape
        x = rearrange(x, 'B D C H W -> (B C) H W D')
        guidance = self.guidance_norm(guidance)
        x = self.swin1(x, guidance)
        x = self.swin2(x, guidance)
        x = rearrange(x, '(B C) H W D -> B D C H W', B=B)
        return x


class ChannelAggregator(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, heads=8, drop_path=0.1, use_linear_transformer=False):
        super().__init__()
        self.heads = heads
        self.use_linear_transformer = use_linear_transformer
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.guidance_norm = nn.LayerNorm(guidance_dim) if guidance_dim > 0 else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _linear_attention(self, q, k, v):
        """
        q, k, v: (BHW, L/S, H, D)
        returns: (BHW, L, H, D)
        """
        # positive feature map
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # KV aggregation
        # (BHW, H, D, D)
        kv = torch.einsum("b s h d, b s h e -> b h d e", k, v)

        # normalizer
        z = 1.0 / (
            torch.einsum("b l h d, b h d -> b l h", q, k.sum(dim=1)) + 1e-6
        )

        # output
        out = torch.einsum("b l h d, b h d e -> b l h e", q, kv)
        out = out * z.unsqueeze(-1)
        return out

    def forward(self, x, guidance):
        # x: (B, D, n_classes, H, W)
        # guidance: BHW, n_classes, D=128
        
        B, D, C, H, W = x.shape

        x = rearrange(x, 'B D C H W -> (B H W) C D')
        guidance = self.guidance_norm(guidance)
        q = self.q(torch.cat([x, guidance], dim=-1))
        k = self.k(torch.cat([x, guidance], dim=-1))
        v = self.v(x)

        q = rearrange(q, 'BHW L (H D) -> BHW L H D', H=self.heads)
        k = rearrange(k, 'BHW S (H D) -> BHW S H D', H=self.heads)
        v = rearrange(v, 'BHW S (H D) -> BHW S H D', H=self.heads)

        if self.use_linear_transformer:
            attn_out = self._linear_attention(q, k, v)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = rearrange(attn_out, 'B L H D -> B L (H D)', H=self.heads)
        x = x + self.drop_path1(attn_out)
        x = self.norm(x)
        x = x + self.drop_path2(self.ffn(x))
        x = rearrange(x, '(B H W) C D -> B D C H W', B=B, H=H, W=W)
        return x

class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, input_resolution, heads=8, use_linear_transformer=False):
        super().__init__()
        self.spatial_agg = SpatialAggregator(hidden_dim=hidden_dim, input_resolution=input_resolution, guidance_dim=guidance_dim, heads=heads)
        self.class_agg = ChannelAggregator(hidden_dim=hidden_dim, guidance_dim=guidance_dim, heads=heads, use_linear_transformer=use_linear_transformer)

    def forward(self, x, text_guidance, vis_guidance):
        B, D, C, H, W = x.shape
        x = self.spatial_agg(x, vis_guidance)
        x = self.class_agg(x, text_guidance)
        return x

class CAFe_DINO(nn.Module):
    def __init__(self, backbone, tokenizer, upsampler, input_resolution, device, dino_dim=1024, aggregator_blocks=6, aggregator_dim=128, pad_len=256, use_linear_transformer=False) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.upsampler = upsampler
        self.device = device
        self.pad_len = pad_len
        self.aggregator_dim=aggregator_dim

        self.corr_embed = nn.Conv2d(1, aggregator_dim, kernel_size=7, stride=1, padding=3)
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(dino_dim, aggregator_dim),
            nn.ReLU(),
        )
        self.vis_guidance_projection = nn.Sequential(
            nn.Conv2d(dino_dim, aggregator_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.aggregator = nn.ModuleList([
            AggregatorLayer(hidden_dim=aggregator_dim, guidance_dim=aggregator_dim, input_resolution=input_resolution, use_linear_transformer=use_linear_transformer) for _ in range(aggregator_blocks)
        ])

        self.reduce_d = nn.Sequential(
            nn.Conv2d(aggregator_dim, aggregator_dim // 2, 1, 1),
            nn.BatchNorm2d(aggregator_dim // 2),
            nn.GELU(),
            nn.Conv2d(aggregator_dim // 2, aggregator_dim // 4, 1, 1),
            nn.BatchNorm2d(aggregator_dim // 4),
            nn.GELU(),
            nn.Conv2d(aggregator_dim // 4, 1, 1, 1)
        )
    
    def get_cost_vol(self, x, text_emb):
        # x: (B, C, H, W)
        # text_emb: (B, T, D)
        B, _, imgH, imgW = x.shape
        H, W, D = imgH // 16, imgW // 16, 1024

        patch_tokens = self.encode_patches(x) # [P, D]
        patch_tokens = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        return torch.einsum('bdhw, btd -> bthw', F.normalize(patch_tokens, dim=1), F.normalize(text_emb.repeat(B, 1, 1), dim=-1)), patch_tokens
    
    def get_corr_embed(self, cost_vol):
        B = cost_vol.shape[0]
        cost_vol = rearrange(cost_vol, 'B C H W -> (B C) H W')  # Stack batch and channel to convolve over each sim map individually
        cost_vol = self.corr_embed(cost_vol.unsqueeze(1))  # (BC, D, H, W)
        cost_vol = rearrange(cost_vol, '(B C) D H W -> B D C H W', B=B)
        return cost_vol
    
    def text_embed(self, class_names) -> torch.Tensor:
        # Build prompts
        prompts = [tpl.format(name) for name in class_names for tpl in PROMPT_TEMPLATES]

        toks = self.tokenizer.tokenize(prompts).to(self.device)

        # Encode and keep patch part
        embs = self.backbone.encode_text(toks)[:, 1024:]  # [C*K, D]

        C = len(class_names)
        K = len(PROMPT_TEMPLATES)
        D = embs.size(1)

        # Group prompts by class and average
        embs = embs.view(C, K, D).mean(dim=1)       # [C, D]

        return F.normalize(embs, p=2, dim=1)


    def build_text_embeddings(self, class_names_list) -> torch.Tensor:
        # Compute once per list item, stack, mean over lists
        stack = torch.stack([
            self.text_embed(class_names)
            for class_names in class_names_list
        ])  # [L, C, D]

        return stack.mean(dim=0)  # [C, D]

    def forward_features(self, x, class_names_list, pre_text_embed):
        # x: (B, C, H, W)
        # text_emb: (B, T, D)
        B, _, imgH, imgW = x.shape
        H, W, D = imgH // 16, imgW // 16, 1024

        # with torch.no_grad():
        if pre_text_embed:
            text_emb = class_names_list
        else:
            text_emb = self.build_text_embeddings(class_names_list)

        cost_vol, patch_tokens = self.get_cost_vol(x, text_emb)  # (B, T, H, W)

        # assert 1 == 0, cost_vol.shape
        cost_vol = self.get_corr_embed(cost_vol)

        text_guidance = self.text_guidance_projection(text_emb)
        text_guidance = repeat(text_guidance.repeat(B, 1, 1), 'B C D -> (B H W) C D', H=cost_vol.shape[-2], W=cost_vol.shape[-1])  # Repeat for each pixel
        vis_guidance = self.vis_guidance_projection(patch_tokens)
        vis_guidance = repeat(vis_guidance, 'B D H W -> (B C) H W D', C=text_guidance.shape[1])
        for i, layer in enumerate(self.aggregator):
            cost_vol = layer(cost_vol, text_guidance, vis_guidance)

        return cost_vol

    def forward(self, x, text_emb, upsample_img=None, pre_text_emb=False):
        B, _, _, _ = x.shape
        # x: (B, D, H, W)
        # text_emb: (B, num_classes, D=1024) if pre-computed (set pre_text_emb=True) otherwise just a class list
        cost_vol = self.forward_features(x, text_emb, pre_text_emb)
        cost_vol = rearrange(cost_vol, 'B D C H W -> B (D C) H W')
        if upsample_img is None:
            hr_features = self.upsampler(x, cost_vol, q_chunk_size=None)  # (B, C, origH, origW)
        else:
            hr_features = self.upsampler(upsample_img, cost_vol, q_chunk_size=None)  # (B, C, hrH, hrW)

        
        hr_features = rearrange(hr_features, 'B (D C) H W -> (B C) D H W', D=self.aggregator_dim)
        hr_cost_vol = self.reduce_d(hr_features)
        hr_cost_vol = rearrange(hr_cost_vol, '(B C) D H W -> B D C H W', B=B).squeeze(1)

        return hr_cost_vol
    
    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        # returns [1, P, D]
        features, patch_tokens, backbone_patch_tokens = self.backbone.encode_image_with_patch_tokens(x)
        return backbone_patch_tokens