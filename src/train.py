import os
import torch
from tokenizer import Tokenizer
from data import get_dataloader
from model import ModelConfig, TransformerModel, generate
import time

def main():
    # Hyperparameters
    batch_size = 16
    seq_len = 128
    d_model = 256
    n_heads = 8
    n_layers = 4
    learning_rate = 3e-4
    max_iters = 1000
    eval_interval = 100
    eval_iters = 20
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tinyshakespeare.txt')
    train_dl = get_dataloader(data_path, tokenizer, seq_len, batch_size, split="train")
    val_dl = get_dataloader(data_path, tokenizer, seq_len, batch_size, split="val")
    
    # Simple iterator wrappers
    train_iter = iter(train_dl)
    val_iter = iter(val_dl)

    def get_batch(split):
        nonlocal train_iter, val_iter
        try:
            if split == "train":
                x, y = next(train_iter)
            else:
                x, y = next(val_iter)
        except StopIteration:
            if split == "train":
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            else:
                val_iter = iter(val_dl)
                x, y = next(val_iter)
        return x.to(device), y.to(device)

    # Init Model
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_heads, # Default to MHA for now, can be changed for GQA
        n_layers=n_layers,
        max_seq_len=seq_len
    )
    
    model = TransformerModel(config).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training Loop
    t0 = time.time()
    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss()
            dt = time.time() - t0
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {dt:.2f}s")
            t0 = time.time()

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training finished. Running generation test...")
    model.eval()
    
    # Generate starting from newline
    context = torch.tensor([tokenizer.encode("\n", allowed_special="all")], dtype=torch.long, device=device)
    generated_idx = generate(model, context, max_new_tokens=200)
    generated_text = tokenizer.decode(generated_idx[0].tolist())
    
    print("\n--- GENERATED TEXT ---")
    print(generated_text)
    print("----------------------")

if __name__ == '__main__':
    main()
