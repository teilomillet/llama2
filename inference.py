import enum
from typing import Optional, List
from pathlib import Path
from numpy import dtype
from sentencepiece import SentencePieceProcessor

from tqdm import tqdm 

import torch
import time
import json


from model import ModelArgs, Transformers

class LLaMA:
    def __init__(self, model: Transformers, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer  # Corrected
        self.args = model_args
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, f"No checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoints {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location="cuda")  # Corrected map_location
            print(f'Loaded checkpoints in {time.time() - prev_time:.2f}s')  # Corrected format
            prev_time = time.time()
            
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())  # Simplified file reading
        
        
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device, 
            **params
        )
        
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfStorage)  # Adjusted tensor type
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)  # Adjusted tensor type
            
        model = Transformers(model_args).to(device)
        
        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {time.time() - prev_time:.2f}s')  # Corrected format
            
        return LLaMA(model, tokenizer, model_args) 
    
    def text_completion(self, prompts: List[str], temperature: float = .6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        # Conversion des prompts en tokens
        prompt_tokens = [self.tokenizer.Encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        
        # assert batch size
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len
        
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        
        # Liste des tokens créés
        pad_id = self.tokenizer.pad_id() 
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device) # Init avec des pad_tokens
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device) # Remplace par des prompt_tokens
            
        eos_reached = torch.tensor([False] * batch_size, device=device)
        
        prompt_tokens_mask = tokens != pad_id # True, si le token est du prompt sinon False
        
        # For loop pour créer les tokens
        for cur_pos in tqdm(range(1, total_len), desc='Créer des tokens...'):
            
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
                
            if temperature > 0: # Appliquée avant le softmax, appuie ou réduit les 'probabilitées'
                probs = torch.softmax(logits[:,-1] / temperature, dim = -1)
                next_token = self._sample_top_p(probs, top_p)
                
            else:
                # Utilise l'approche 'Greedy' (token avec la meilleure 'probabilitée')
                next_token = torch.argmax(logits[:, -1], dim = -1)
            
            next_token = next_token.reshape(-1)
            
            # Remplace le padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            
            # End Of Sequence (EOS)
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break
            
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Coupe le prompt si EOS est présent.
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.Decode(current_prompt_tokens))
        return(out_tokens, out_text)
        
        
    def _sample_top_p(self, probs, p):
        '''
        Classe et arrange les tokens par top 'probabilités'
        '''
        probs_sort, probs_idx = torch.sort(probs, dim = -1, descending= True)
        probs_sum = torch.cumsum(probs_sort, dim= -1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        
        # Redistribution des probabilités
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token) # Parce qu'on les a classés, il faut les rapprocher avec l'index.
        return next_token


    
if __name__ == '__main__':
    torch.manual_seed(0)
    
    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    
    prompts = [
        "Comment faire une mayonnaise ?",
        "How to do a mayonnaise ?",
        "If my aunt would have some, it would be "
    ]
    
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )
    
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-'*50)