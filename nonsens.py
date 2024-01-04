import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096 # dimension de l'embeddings // dépend de la taille du model. (ici base model)
    n_layers: int = 32
    n_heads: int = 32 # Nombre de tête pour les requêtes/queries (q)
    n_kv_heads: Optional[int] = None # Nombre de tête pour les clefs/keys (k) et les valeurs/values (v)
    vocab_size: int = -1 # Change avec le tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5 # Besoin pour la normalisation (évite de diviser par 0)
    
    # KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: Optional[str] = None
    
    
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta_scalar: float = 10000.0):
    '''
    Rotary Position Embedding (RoFormer) 
    
    https://arxiv.org/abs/2104.09864 
    
    L'input de cette function est la sequence maximal * 2.
    Precompute tous les m et theta possible pour cette sequence len.
    '''
    
    assert head_dim % 2 == 0, "Dimension de l'embeddings doit être paire."
    
    # Build the theta parameters (sequence) 
    # theta_i = theta_scalar ^ (-2(i-1)/dim) for i = [1,2, ... dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float().to(device)
    theta = 1.0 / (theta_scalar ** (theta_numerator / head_dim)) # (head_dim / 2)
    
    m = torch.arange(seq_len, device=device)
    
    # Mutilplie chaque theta par chaque position (m) en utilisant l'outer product
    freqs = torch.outer(m, theta).float() # (seq_len) outer product* (head_dim /2) -> (seq_len, head_dim / 2)
    
    # Compute des nombre complexe dans leur forme polaire c = R * exp(i * m * theta), where R = 1 :
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    '''
   
    Splitting les vecteurs embeddings en de nouveau tensors, de facon à ce qu'ils aient la moitiés de la dimension. 
    On le fait en groupant 2 dimensions consécutives. Puis on le transforme à l'aide de nombre complexe.
    Ensuite on le multiplie avec (m, theta).
    
    '''
    
    # Prend 2 dimensions consécutives et groupe les ensemble, puis transforme la un tensor complexe (view_as_complex).
    # (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:1], -1, 2))
    
    # (seq_len, head_dim /2 ) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # (B, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim/2) => (B, sq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    
    # (B, sq_len, h, head_dim / 2) -> (B, sq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    
    # Applati -> (B, sq_len, h, head_dim / 2, 2) -> (B, sq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    '''
    Duplique les têtes KV.
    '''
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    
    if n_rep == 1:
        return x
    else:
        # (B, seq_len, N_KV_head, 1, head_dim)
        return (
            x[:, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class RMSNorm(nn.Module):
    
    '''
    Normalisation focus sur le scaling.
    
    https://arxiv.org/abs/1910.07467

    '''
 
    def __init__(self, dim: int, eps:float = 1-6):
        super().__init__()
        
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(dim)) # Gamma parameter
        
    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim) * (B, seq_len, 1) => (B, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor ):
        # (dim) * (B, seq_len, dim) * (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
class SelfAttention(nn.Module):
    
    '''
    J'ai qu'un seul GPU donc pas besoin de parallelisme. Ce qui simplifie le code.
    '''
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads # Nombre de tête pour (kv)
        self.n_heads_q = args.n_heads # Nombre de tête pour (q)
        self.n_rep = self.n_heads_q // self.n_kv_heads # Ratio nombre de tête (q) et nombre de tête (kv), pour dupliquer KV -> Q.
        self.head_dim = args.dim // args.n_heads # Indique la dimension de chaque tête
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # Caching K & V
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape 
        # (B, 1, dim) = (B = Batch, seq_len = 1, dim)
        
        # Application des poids (Wq, Wk, Wv) aux queries (q), keys (k) et values (v)
        
        # (B, 1 dim) -> (B, 1, H_Q * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_KV * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (B, 1, H_Q * head_dim) -> (B, 1, H_Q, head_dim)
        # Division en H heads.
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        
        # (B, 1, H_KV * head_dim) -> (B, 1, H_KV, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Application du RoFORMER (Rotary Position Embeddings)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        # Remplace le cache du token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        # Extrait les keys et values qui sont dans le cache
        # (B, seq_len_KV, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]
        
        # Duplique K et V heads afin d'avoir le même nombre de tête que Q (as done by Meta)
        keys = repeat_kv(keys, self.n_rep)        
        values = repeat_kv(values, self.n_rep)
        
        # (B, 1, H_Q, head_dim) -> (B, H_Q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # (B, H_Q, 1, head_dim) @ (B, H_Q, head_dim, seq_len_KV) -> (B, H_Q, 1, seq_len_KV)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=1).type_as(xq)
        
        # (B, H_Q, 1, seq_len) @ (B, H_Q, seq_len, head_dim) -> (B, H_Q, 1, head_dim)
        output = torch.matmul(scores, values)
        
        # (B, H_Q, 1, head_dim) -> (B, 1, H_Q, head_dim) -> (B, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, dim)
        
class FeedForward(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        # Arondi au plus proche multiple du parametres de mulitples
        hidden_dim = args.multiple_of * ((hidden_dim * args.multiple_of -1 ) // args.multiple_of)
        
        # Ws for SwiGLU activation function
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
            
    
class EncoderBlock(nn.Module):
    
    '''
    Corps du transformer. 
    
    '''
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # Normalisations
        
        # Avant self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Avant feed forward block
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
        
        
        
        
 
class Transformers(nn.Module):
    '''
    Llama architecture.
    Les noms sont quasiment les mêmes que ceux de llama.
    '''
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1, "Besoin d'un vocab size."
        
        self.args = args
        self.vocab_size =args.vocab_size
        self.n_layers = args.n_layers
        
        # Input embeddings, converti les inputs en embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim) 
        
        # Le passe dans une liste de layers
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        # L'output du dernier layer est normalisé (RMSNorm) avant d'etre 'output'
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, 
                                                              self.args.max_seq_len * 2, 
                                                              device=self.args.device)
      
    def forward(self, tokens: torch.Tensor, start_pos: int):
        '''
        
        Code pour l'inférence et non de l'entrainement du model. 
        Ici on process 1 token à la fois, c'est pourquoi on utilise le KV cache.
        Et que l'on utilise le model llama pré-entrainé.
        
        '''
        
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Process 1 token à la fois"
        
        # (B, seq_len) -> (B, seq_len, dim) Token -> Embeddings
        h = self.tok_embeddings(tokens)
        
        # Rotary Position Embeddings implementation
        # Retrouver le paire (m, theta) correspondant à la position de l'embeddings [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]
        
        # Applique aux prochains layers de l'encodeur
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        
        output = self.output(h).float()
        return output
        