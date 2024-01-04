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
    n_kv_hedas: Optional[int] = None # Nombre de tête pour les clefs/keys (k) et les valeurs/values (v)
    vocab_size: int = -1 # Change avec le tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5 # Besoin pour la normalisation (évite de diviser par 0)
    
    # KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None
    
    
def precompute_theta_pos_frequencies(head_dim:int, seq_len: int, device: str, theta: float = 10000.0):
    '''
    Rotary Position Embedding (RoFormer) 
    
    https://arxiv.org/abs/2104.09864 
    
    L'input de cette function est la sequence maximal * 2.
    Precompute tous les m et theta possible pour cette sequence len.
    '''
    
    assert head_dim % 2 == 0, "Dimension de l'embeddings doit être paire."
    
    # Build the theta parameters (sequence) 
    # theta_i = 10_000 ^ (-2(i-1)/dim) for i = [1,2, ... dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (head_dim / 2)
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
 
class Transformers(nn.Module):
    '''
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
        