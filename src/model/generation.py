import torch
from torch.nn import functional as F

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressive generation loop.
    idx: (B, T) LongTensor
    """
    model.eval()
    for _ in range(max_new_tokens):
        # crop context
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    model.train()
    return idx
