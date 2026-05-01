"""
LorentzMultiheadAttention module implements multi-head attention in Lorentzian geometry.
It supports both full attention (hyperbolic self attention) and linear focused attention.

Based on:
    - Hypformer: Exploring Efficient Hyperbolic Transformer Fully in Hyperbolic Space (https://arxiv.org/abs/2407.01290)
    - Fully Hyperbolic Neural Networks (https://arxiv.org/abs/2105.14686)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear import LorentzLinear
from .cls import LorentzCLS
from .conv_util_layers import LorentzLayerNorm, LorentzActivation, LorentzDropout, LorentzNormalization, LResNet
from ..manifold.lorentz import Lorentz

class LorentzMultiheadAttention(nn.Module):
    """
    Multi-head attention mechanism in Lorentzian (hyperbolic) geometry.

    Requires input dimension to be ***dimension per head***

    Args:
        manifold (Lorentz): Lorentzian manifold object for geometry operations.
        in_channels (int): Input dimensionality.
        out_channels (int): Output dimensionality.
        num_heads (int): Number of attention heads.
        use_weight (bool): Whether to use a trainable value projection (Wv).
        power_k (float): Exponent used in the linear focused approximation.
        attention_type (str): Either 'full' (self-attention) or 'linear_focused'.
        trans_heads_concat (bool): Whether to concatenate attention heads and linearly transform output.
        normalize (bool): Whether to normalize input queries and keys.
    """
    def __init__(self, manifold: Lorentz,  in_channels, out_channels, num_heads, use_weight=True, power_k=2.0, attention_type='linear_focused', trans_heads_concat=False, normalize=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = attention_type
        self.Wk = nn.Linear(num_heads * self.in_channels, num_heads * (self.out_channels - 1))
        self.Wq = nn.Linear(num_heads * self.in_channels, num_heads * (self.out_channels - 1))

        if use_weight:
            self.Wv = nn.Linear(num_heads * self.in_channels, num_heads * (self.out_channels - 1))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(num_heads * out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        if self.attention_type=='linear_focused':
            self.v_map_mlp = nn.Linear(in_channels - 1, out_channels, bias=True)
            self.norm_scale = nn.Parameter(torch.ones(()))
        self.power_k = power_k
        self.trans_heads_concat = trans_heads_concat
        if self.trans_heads_concat:
            self.final_linear = nn.Linear(self.num_heads * (self.out_channels), self.num_heads * self.out_channels - 1) 
        self.normalize = normalize

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def project(self, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def shape_mask(self, mask, batch_size: int, num_heads: int, seq_len: int):
        if mask is None:
            return None
        if mask.dim() == 4:
            if mask.shape == (batch_size, num_heads, seq_len, seq_len):
                return mask
            else:
                raise ValueError(f"Mask with 4 dims must be [B, H, N, N]")

        if mask.dim() == 2:
            m, n = mask.shape
            # [N, N]
            if (m == seq_len) and (n == seq_len):
                mask = mask.unsqueeze(0).unsqueeze(0)
            else: # [B, N]
                mask = mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, N]
            return mask

        if mask.dim() == 3:
            b, m, n = mask.shape
            # [B, N, N]
            if (m == seq_len) and (n == seq_len):
                mask = mask.unsqueeze(1)  # [B, 1, N, N]
                return mask

            # [B, 1, N]
            if (m == 1) and (n == seq_len):
                mask = mask.squeeze(1)
                mask = mask.unsqueeze(1).unsqueeze(2)
                return mask

            # [B, H, N]
            if (m == num_heads) and (n == seq_len):
                mask = mask.unsqueeze(2)
                return mask
    
    def apply_rotary_embeddings(self, x, freqs_complex, device):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freqs_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)

    def full_attention(self, qs, ks, vs, output_attentions=False, mask=None):
        """
        Computes Lorentz full attention via hyperbolic inner products and centroid.

        Returns:
            attention_output (Tensor): Resulting attended output.
            att_weight (Tensor, optional): Attention weight matrix if output_attentions=True.
        """
        # reshape the inputs
        qs = self.project(qs)
        ks = self.project(ks)
        vs = self.project(vs)
        # normalize input
        if self.normalize:
            qs = LorentzNormalization(self.manifold)(qs)
            ks = LorentzNormalization(self.manifold)(ks)
        # negative squared distance (less than 0)
        att_weight = 2 * self.manifold.c + 2 * self.manifold.cinner(qs.transpose(1, 2), ks.transpose(1, 2))  # [B, H, N, N]
        att_weight = att_weight / self.scale + self.bias  # [B, H, N, N]
        if mask is not None:
            att_weight = att_weight.masked_fill(mask, -1e18)
        att_weight = nn.Softmax(dim=-1)(att_weight)  # [B, H, N, N]
        att_output = self.manifold.lorentzian_centroid(vs.transpose(1, 2), att_weight)  # [B, H, N, D]
        att_output = att_output.transpose(1, 2) # [B, N, H, D]
        if self.trans_heads_concat:
            att_output_space = self.final_linear(att_output.reshape(att_output.size(0), att_output.size(1), self.num_heads * self.out_channels))
            att_output_time = ((att_output_space**2).sum(dim=-1, keepdims=True) + self.manifold.c).sqrt()
            att_output = torch.cat([att_output_time, att_output_space], dim=-1)     
            att_output = att_output       
        else:
            att_output = self.manifold.lorentzian_centroid(att_output)
        if output_attentions:
            return att_output, att_weight
        else:
            return att_output

    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attentions=False, mask=None):
            """
            Computes linear focused attention in Lorentz geometry.
            """
            qs = hyp_qs[..., 1:]
            ks = hyp_ks[..., 1:]
            v = hyp_vs[..., 1:]
            phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [B, N, H, D]
            phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [B, N, H, D]

            phi_qs = self.fp(phi_qs, p=self.power_k)  # [B, N, H, D]
            phi_ks = self.fp(phi_ks, p=self.power_k)  # [B, N, H, D]

            k_transpose_v = torch.einsum('bnhm,bnhd->bhmd', phi_ks, v)  # [B, H, D, D]
            numerator = torch.einsum('bnhm,bhmd->bnhd', phi_qs, k_transpose_v)  # [B, N, H, D]
            denominator = torch.einsum('bnhd,bhd->bnh', phi_qs, torch.einsum('bnhd->bhd', phi_ks))  # [B, N, H]
            denominator = denominator.unsqueeze(-1)  #
            attn_output = numerator / (denominator + 1e-6)  # [B, N, H, D]
            vss = self.v_map_mlp(v)  # [B, N, H, D]
            attn_output = attn_output + vss  # preserve its rank, [B, N, H, D]

            if self.trans_heads_concat:
                attn_output = self.final_linear(attn_output.reshape(attn_output.size(0), -1, self.num_heads * self.out_channels))
            else:
                attn_output = attn_output.mean(dim=1)

            attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
            attn_output = torch.cat([attn_output_time, attn_output], dim=-1)
            if output_attentions:
                return attn_output, attn_output
            else:
                return attn_output

    def forward(self, query_input, source_input, output_attentions=False, mask=None, rot_pos=None, edge_index=None, edge_weight=None):
        """
        Forward pass for Lorentz multi-head attention.

        Args:
            query_input (Tensor): Input query sequence of shape [B, N, D].
            source_input (Tensor): Input key/value source sequence.
            output_attentions (bool): If True, returns attention weights.
            mask (Tensor): Attention mask.
            rot_pos (Tensor): Rotary position encodings.

        Returns:
            final_output (Tensor): Attention output of shape [B, N, D].
            attn (Tensor, optional): Attention weights.
        """
        batch_size, seq_length, embed_dim = source_input.size()
        if mask is not None:
            mask = self.shape_mask(mask, batch_size, self.num_heads, seq_length)
        query = self.Wq(query_input).view(batch_size, seq_length, self.num_heads, self.out_channels - 1) # [B, N, H, D]
        key = self.Wk(source_input).view(batch_size, seq_length, self.num_heads, self.out_channels - 1) # [B, N, H, D]
        if rot_pos is not None:
            query = self.apply_rotary_embeddings(query, rot_pos, query.device)
            key = self.apply_rotary_embeddings(key, rot_pos, key.device)
        if self.use_weight:
            value = self.Wv(source_input).view(batch_size, seq_length, self.num_heads, self.out_channels - 1) # [B, N, H, D]
        else: 
            value = source_input.view(batch_size, seq_length, self.num_heads, self.out_channels) # [B, N, H, D]
        if output_attentions:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(
                    query, key, value, output_attentions, mask)  # [B, N, H, D]
            elif self.attention_type == 'full':
                attention_output, attn = self.full_attention(
                    query, key, value, output_attentions, mask)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(
                    query, key, value, output_attentions, mask)  # [B, N, H, D]
            elif self.attention_type == 'full':
                attention_output = self.full_attention(
                    query, key, value, output_attentions, mask)
            else:
                raise NotImplementedError


        final_output = attention_output

        if output_attentions:
            return final_output, attn
        else:
            return final_output