import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_sliding_window_mask(seq_len, window_size, device):
    """Efficiently generates a sliding window mask."""
    arange = torch.arange(seq_len, device=device)
    q_idx = arange.view(-1, 1)  # (L, 1)
    kv_idx = arange.view(1, -1)  # (1, L)
    mask = (torch.abs(q_idx - kv_idx) <= window_size).to(torch.float32)  # (L, L)
    return mask

def relative_positional(score, q_idx, kv_idx, scaling_factor=1.0, tau=1.0):
    """Adds relative positional bias to attention scores."""
    relative_dist = (q_idx - kv_idx).to(torch.float32)/scaling_factor
    relative_positions = torch.exp(-torch.abs(relative_dist)/tau)  # Exponential decay
    return score - relative_positions  # Apply relative position bias

class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)
        self.scaling_factor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, Q, K, V, mask=None):
        """Sliding window attention forward pass.
        
        Args:
            Q: Query tensor (B, H, L, d_q).
            K: Key tensor (B, H, L, d_k).
            V: Value tensor (B, H, L, d_v).
            mask: input attention mask (B, 1, L).

        Returns:
            output: Attention tensor (B, H, L, d_v).
            attention: Attention weights tensor (B, H, L, L).
        """

        d_k = Q.shape[-1]
        L = Q.shape[-2]
        device = Q.device
        
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, L, L)
        # Non-causal mask for sliding window (half on each side)
        window_mask = create_sliding_window_mask(L, self.window_size//2, device)  # (L, L)
        
        # Combine input mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask.expand(-1, -1, -1, L) #  (B, 1, L, L)
            combined_mask = mask * window_mask.unsqueeze(0).unsqueeze(0)  # (B, H, L, L)
        else:
            combined_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        
        scores = scores.masked_fill(combined_mask == 0, -1e9)
        # Add relative positional bias
        q_idx = torch.arange(L, device=device).view(1, 1, L, 1)
        kv_idx = torch.arange(L, device=device).view(1, 1, 1, L)
        scores = relative_positional(scores, q_idx, kv_idx, self.scaling_factor, self.tau)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = attention @ V  # (B, H, L, d_v)
        return output, attention
    


class ScaledDotProductAttention(nn.Module):
    '''Implements basic attention mechanism.
    '''
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        '''Forward pass.
        Args:
            Q: Query tensor (B, n_heads, L, d_q).
            K: Key tensor (B, n_heads, L, d_k).
            V: Value tensor (B, n_heads, L, d_v).
            mask: Mask tensor.
        Returns:
            output: Attention tensor (B, n_heads, L, d_v).
            attention: Attention weights tensor (B, n_heads, L, L).
        '''
        d_k = Q.size(-1)
        L = Q.size(-2)
        scores = Q@K.transpose(-2,-1)/math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask.expand(-1, -1, -1, L) #  (B, 1, L, L)
            scores = scores.masked_fill(mask==0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = attention@V
        return output, attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_dim, d_model, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.W_o = nn.Linear(n_heads*head_dim, d_model, bias=False)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        B = q.size(0)
        residual = q
        Q = self.W_q(q).view(B, -1, self.n_heads, self.head_dim).transpose(1,2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.head_dim).transpose(1,2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.head_dim).transpose(1,2)
        if mask is not None:
            mask = mask.unsqueeze(1)    # add dimension for n_heads
        output, attention = self.attention(Q, K, V, mask)   # output (B, n_heads, L, head_dim)
        output = output.transpose(1,2).contiguous().view(B, -1, self.n_heads*self.head_dim)
        output = self.dropout(self.W_o(output))
        output = self.layer_norm(output + residual)
        return output, attention
    
class MultiHeadSWAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, window_size=1024,dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads*d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads*d_v, bias=False)
        self.W_o = nn.Linear(n_heads*d_v, d_model, bias=False)
        self.attention = SlidingWindowAttention(window_size=window_size, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """Multi-head attention forward pass.

        Args:
            q: Query tensor (B, L, d_model).
            k: Key tensor (B, L, d_model).
            v: Value tensor (B, L, d_model).
            mask: input attention mask (B, L).

        Returns:
            output: Attention tensor (B, L, d_model).
            attention: Attention weights tensor (B, n_heads, L, L).
        """

        B, _, _ = k.shape

        residual = q
        Q = self.W_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.d_v).transpose(1,2)
        if mask is not None:
            mask = mask.unsqueeze(1)    # add dimension for n_heads
        output, attention = self.attention(Q, K, V, mask)   # output (B, n_heads, L, head_dim)
        output = output.transpose(1,2).contiguous().view(B, -1, self.n_heads*self.d_v)
        output = self.dropout(self.W_o(output))
        output = self.layer_norm(output + residual)
        return output, attention

        

