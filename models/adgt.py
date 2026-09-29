"""
ADGT: parallel GCN-Transformer for 2D-to-3D human pose lifting.

Paper components -> modules in this file:
    HS-GCN  (Hop-wise Scalable GCN, Sec. 3.3)          -> HSGCN
    RET     (Register-based Enhancement, Sec. 3.4)     -> RET
    ALFE    (Attention-based Local Feature Extractor)  -> ALFE
    fusion + skip connection (Eqs. 5-6)                -> ADGTLayer
    full network (Eqs. 1, 7)                           -> ADGT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, unpack, repeat
try:
    from timm.layers import DropPath
except ImportError:  # older timm
    from timm.models.layers import DropPath

# Human3.6M 16-joint skeleton (Hip, RHip, RKnee, RFoot, LHip, LKnee, LFoot, Spine,
# Thorax, Head, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist).
H36M_EDGES = [[0, 1], [1, 2], [2, 3],
              [0, 4], [4, 5], [5, 6],
              [0, 7], [7, 8], [8, 9],
              [8, 10], [10, 11], [11, 12],
              [8, 13], [13, 14], [14, 15]]


def hop_adjacency(edges, hops, num_joints=16, normalize=True):
    """Per-hop adjacency stack of shape (1, len(hops), N, N).

    Entry [k] connects each joint to the joints exactly hops[k] edges away (plus a
    self-loop), optionally symmetrically normalised as D^-1/2 A D^-1/2.
    """
    adj = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1.0

    # Shortest-path distances by repeated frontier expansion (the skeleton is a tree).
    dist = torch.full((num_joints, num_joints), float("inf"))
    dist.fill_diagonal_(0)
    reach = torch.eye(num_joints)
    for step in range(1, num_joints):
        reach = ((reach @ adj) > 0).float()
        dist[(reach > 0) & torch.isinf(dist)] = step

    stack = torch.stack([(dist == h).float() + torch.eye(num_joints) for h in hops])
    if normalize:
        d_inv_sqrt = stack.sum(-1).pow(-0.5).unsqueeze(-1)
        stack = d_inv_sqrt * stack * d_inv_sqrt.transpose(-1, -2)
    return stack.unsqueeze(0)


class MLP(nn.Module):
    """Pre-LayerNorm two-layer MLP."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features * 2
        out_features = out_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.activation1(self.fc1(self.norm(x))))
        return self.drop2(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class RET(nn.Module):
    """Register-based Enhancement for Transformers (Eqs. 14-15).

    Learnable register tokens are concatenated to the joint tokens for attention and
    discarded afterwards, so they absorb global information without being forwarded.
    """

    def __init__(self, dim, num_heads, num_joints, num_registers=8, mlp_ratio=4.0,
                 mlp_drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim))
        self.registers = nn.Parameter(torch.randn(num_registers, dim))
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=mlp_drop)

    def forward(self, x):
        B = x.shape[0]
        x = x + self.Spatial_pos_embed
        registers = repeat(self.registers, "n d -> b n d", b=B)
        x, ps = pack([x, registers], "b * d")
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x, _ = unpack(x, ps, "b * d")  # drop the register tokens
        return x + self.drop_path(self.mlp(self.norm2(x)))


class HSGCN(nn.Module):
    """Hop-wise Scalable GCN (Eqs. 8-13).

    One graph convolution per hop distance (weights W_k, Eq. 8); the hop features are
    then re-weighted per joint by scaling factors alpha = softmax_k(sum_n <H_hop, H_in>)
    (Eqs. 10-12) and summed over hops (Eq. 13).
    """

    def __init__(self, dim, hops=(1, 2, 3), num_joints=16, edges=H36M_EDGES, proj_drop=0.0):
        super().__init__()
        self.register_buffer("adjs", hop_adjacency(edges, hops, num_joints), persistent=False)
        self.linears = nn.Parameter(torch.empty(1, len(hops), dim, dim))
        nn.init.xavier_normal_(self.linears)
        self.bias = nn.Parameter(torch.zeros(1, len(hops), num_joints, dim))
        self.norm_hf = nn.LayerNorm(dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.unsqueeze(1)                                                    # B,1,N,C
        hop_f = self.proj_drop(self.adjs @ self.norm_hf(x) @ self.linears + self.bias)  # B,K,N,C
        alpha = (hop_f @ x.transpose(-2, -1)) * hop_f.shape[-1] ** -0.5       # B,K,N,N
        alpha = F.softmax(alpha.sum(-1, keepdim=True), dim=1)                 # B,K,N,1
        return (alpha * hop_f).sum(dim=1)


class ALFE(nn.Module):
    """Attention-based Local Feature Extractor (Eqs. 16-17).

    The global embedding forms the query and the local embedding the key/value, so the
    global features select the local information they need.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)

    def _split(self, x):
        B, N, _ = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, global_f, local_f):
        B, N, C = local_f.shape
        q = self._split(self.w_q(self.norm_q(global_f)))
        k = self._split(self.w_k(self.norm_k(local_f)))
        v = self._split(local_f)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.head_dim ** -0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.linear(self.mlp_norm(out)) + out


class Fusion(nn.Module):
    """Fuses the extracted local features with the global embedding."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm_proj = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.project_drop = nn.Dropout(dropout)

    def forward(self, local_ex, global_f):
        fused = local_ex + global_f
        return self.project_drop(self.activation(self.projection(self.norm_proj(fused)) + fused))


class ADGTLayer(nn.Module):
    """One ADGT layer: HS-GCN and RET run in parallel on the same hidden state."""

    def __init__(self, dim, num_joints, num_heads, num_registers, ca_heads, mlp_ratio,
                 dropout, attn_drop, drop_path, hops=(1, 2, 3), edges=H36M_EDGES):
        super().__init__()
        self.mpnn = HSGCN(dim, hops, num_joints, edges, proj_drop=dropout)
        self.attn = RET(dim, num_heads, num_joints, num_registers, mlp_ratio,
                        mlp_drop=dropout, attn_drop=attn_drop, drop_path=drop_path)
        self.cross_attention = ALFE(dim, ca_heads)
        self.data_aggregation = Fusion(dim, dropout)

    def forward(self, x):
        global_f = self.attn(x)
        local_f = self.mpnn(x)
        return self.data_aggregation(self.cross_attention(global_f, local_f), global_f) + x


class ADGT(nn.Module):
    """2D joints (B, N, 2) -> 3D joints (B, N, 3)."""

    def __init__(self, dim_model=96, n_layer=5, n_pts=16, n_head=4, n_register=8, h_ca=4,
                 mlp_ratio=4, dropout=0.1, attn_drop=0.0, drop_path=0.1,
                 hops=(1, 2, 3), edges=H36M_EDGES):
        super().__init__()
        self.gcn_encoder = nn.Linear(2, dim_model)
        self.gps = nn.ModuleList([
            ADGTLayer(dim_model, n_pts, n_head, n_register, h_ca, mlp_ratio,
                      dropout, attn_drop, drop_path, hops, edges)
            for _ in range(n_layer)])
        self.gcn_decoder = nn.Linear(dim_model, 3, bias=False)

    def forward(self, x):
        x = self.gcn_encoder(x)
        for layer in self.gps:
            x = layer(x)
        return self.gcn_decoder(x)
