import torch


def srgb_to_linear(c):
    a = 0.055
    return torch.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def linear_to_srgb(c):
    a = 0.055
    c = torch.clamp(c, 0.0, 1.0)  # simple gamut clamp in linear light
    return torch.where(c <= 0.0031308, 12.92 * c, (1 + a) * torch.pow(c, 1 / 2.4) - a)


def oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    R = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    G = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    B = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return torch.stack([R, G, B], dim=-1)


def oklch_grid(h_k, w_k, col_range=.7):
    i, j = torch.meshgrid(
        torch.arange(-col_range / 2, col_range / 2, col_range / h_k),
        torch.arange(-col_range / 2, col_range / 2, col_range / w_k),
        indexing='ij'
    )
    rgb = oklab_to_linear_srgb(torch.full_like(i, .7), i, j)
    return rgb


def visualize_attention_oklab(attn, h_q, w_q, h_k=None, w_k=None):
    h_k = h_k or h_q
    w_k = w_k or w_q

    num_q, num_k = attn.shape
    assert 0 < h_q * w_q <= num_q
    assert 0 < h_k * w_k <= num_k
    if h_q * w_q < num_q: attn = attn[-h_q * w_q:]
    if h_k * w_k < num_k: attn = attn[:, -h_k * w_k:]

    # rows sum to 1
    attn = torch.nn.functional.normalize(attn, p=1, dim=1)

    ref_lin = oklch_grid(h_k, w_k).to(attn.device)  # [h_k, w_k, 3]
    ref_rgb = linear_to_srgb(ref_lin)
    ref_lin = ref_lin.view(-1, 3)

    out_lin = attn @ ref_lin  # [(h_q*w_q), 3]
    out_rgb = linear_to_srgb(out_lin.view(h_q, w_q, 3))
    return ref_rgb, out_rgb
