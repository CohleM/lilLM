import math
import torch
from torch import nn



def data_loader(data_path, split):
    filename = os.path.join(data_path, f'{split}.bin')
    data = np.memmap(filename, mode='r')
    ids = torch.randint((len(data)-block_size), (batch_size,))

    X = torch.stack([ torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ids])
    Y = torch.stack([ torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ids])

    if device_type=='cuda':
        # pin to (page-locked) memory in the host (CPU) memory, so the write to GPU is faster, also enable async data transfer(non_blocking=True)
        return X.pin_memory().to(device,non_blocking=True), Y.pin_memory().to(device,non_blocking=True)
    else:
        return X.to(device), Y.to(device)



def precompute_freq_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) 
    freqs = torch.outer(t, freqs).float()  
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_pe(xq, xk, cis):
    
    # at this point cis will have shape (T,head_dim//2) and we transform it to (1,T,1,head_dim//2)
    #cause we need to broadcast it so we can multiply it with (B,T,n_heads, head_dim//2)
    # head_dim is div by 2 cause, it is complex form (ex. 1+2j)
    def reshape_for_broadcast(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1]) # x.shape = (B,T,n_heads,head_dim), T and C should batch
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    
    # (B,T,n_heads,head_dim) --> (B,T,n_heads, head_dim//2, 2), cause view_as_complex expects last dim to be 2. [1,2] --> [1+2j]
    xq_ = torch.view_as_complex(xq.view(*xq.shape[:-1],-1,2)) 
    xk_ = torch.view_as_complex(xk.view(*xk.shape[:-1],-1,2))
    
    print(cis.shape, xq_.shape, xk_.shape)
    
    cis = reshape_for_broadcast(cis,xq_)
    
    
    
    # convert to real i.e (B,T,n_heads, head_dim//2,2) --> (B,T,n_heads, head_dim)
    xq_out,xk_out = torch.view_as_real(xq_*cis).flatten(3), torch.view_as_real(xk_*cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
    

def repeat_kv(x, n_rep):
    B,T,kv_heads, head_dim = x.shape
    if n_rep==1:
        return x
    else:
        return (
                x[:,:,:,None,:]
                .expand(size=(B,T,kv_heads,n_rep, head_dim))
                .reshape(B,T,kv_heads * n_rep, head_dim)
               )


class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cfg = config
        self.head_dim = config.d_model // config.q_heads
        
        self.wq = nn.Linear(config.d_model, config.q_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(config.q_heads * self.head_dim, config.d_model, bias=False)
        
    
        self.cache_k = None
        self.cache_v = None
        
        self.attn_dropout = nn.Dropout(config.dropout)
        mask = torch.full((1,1,config.max_seq_len, config.max_seq_len), float('-inf')) # fill all elements by -inf
        mask = torch.triu(mask, diagonal=1) # only keep the values of the upper triangular matrix, others to 0
        self.register_buffer('mask', mask, persistent=False) # register buffer, i.e not trainable parameter, also don't save in state dict
        
        

    def forward(self,x,start_pos, freq_cis):
        
        B,T,C = x.shape # (batch_size, seq_len, emb_dim)
        
        # project input(self) into Q,K,V (this is why it's called self-attention)
        xq,xk,xv = self.wq(x), self.wk(x), self.wv(x) 
        
        # divide into their respective heads, (B,T,C) --> (B,T,q_heads or kv_heads,head_dim)
        xq = xq.view(B,T,self.cfg.q_heads, self.head_dim)
        xk = xk.view(B,T,self.cfg.kv_heads, self.head_dim)
        xv = xv.view(B,T,self.cfg.kv_heads, self.head_dim)


        xq,xk = apply_rotary_pe(xq,xk, freq_cis)
        
        # use kv cache
        if not self.training:
            if self.cache_k is None and self.cache_v is None:
                self.cache_k = torch.zeros(
                    self.cfg.max_batch_size,
                    self.cfg.max_seq_len,
                    self.cfg.kv_heads,
                    self.head_dim,
                    device=x.device
                )
                self.cache_v = torch.zeros(
                    self.cfg.max_batch_size,
                    self.cfg.max_seq_len,
                    self.cfg.kv_heads,
                    self.head_dim,
                    device=x.device 
                )
            
            #update the cache, i.e append the new sequence to where we left of (start_pos) in the last iteration.
            #we add :B and not only : because B != max_batch_size

            self.cache_k[:B, start_pos: start_pos + T] = xk # T is most probably 1, cause during inference we only process on last tokens
            self.cache_v[:B, start_pos: start_pos + T] = xv
            
            xk = self.cache_k[:B, :start_pos + T]
            xv = self.cache_v[:B, :start_pos + T]
        
        # repeat key and value matrices, for group query attention
        xk = repeat_kv(xk, self.cfg.q_heads // self.cfg.kv_heads)
        xv = repeat_kv(xv, self.cfg.q_heads // self.cfg.kv_heads)


        xq,xk,xv = xq.transpose(1,2), xk.transpose(1,2), xv.transpose(1,2) #transpose T,n_head cause we perform matrix mul on last two dims
        
        attn_score = xq @ xk.transpose(-1,-2)/ math.sqrt(self.head_dim)
        
        attn_score = attn_score + self.mask[:,:,:T,:T] # cause its inputs don't always have max_seq_len 
        attn_score = torch.softmax(attn_score, dim=-1) #normalize and make -inf to 0
        attn_score = self.attn_dropout(attn_score) 
        out = attn_score @ xv
        out = out.transpose(1,2).contiguous().view(B,T,-1)
        
        out = self.wo(out)
        
        return out





class FFN(nn.Module):
    def __init__(self,d_model, hidden_dim, multiple_of, dropout):
        super().__init__()
        self.d_model = d_model
        if hidden_dim is None:
            hidden_dim = 4*d_model
            hidden_dim = int(2/3 * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) # make hidden dim multiple of {multiple_of}
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False) # W
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False) # W2
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False) # V

        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        # Swish(x) = x * sigmoid(ßx)
        # GLU(x) = sigmoid(xW+b)⊗(xV+c)
        # SwiGLU(x) = Swish(Wx + b) ⊗ (Vx+c)
        # Omit bias -> SwiGLU(x) = Swish(xW) ⊗ (xV)
        # FFFNswiglu = SwiGLU(x)W2, with b parameter=1
        # since there is additional parameter V when compared to what's used in transformers, 
        # we reduce the output dimension of W,V by 2/3 and input dimension of W2 by 2/3
        
        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))



class RMSNorm(nn.Module):
    def __init__(self,d_model,norm_eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.eps = norm_eps
    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
    def forward(self,x):
        # convert to higher precision float32 for rms for accuracy, then back to their original type
        out = self._norm(x.float()).type_as(x) 
        return self.gain * out



