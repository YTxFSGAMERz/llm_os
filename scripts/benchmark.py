import time
import torch
from src.model import TransformerModel, ModelConfig
from src.model.generation import generate

def measure_forward_pass(model, batch_size, seq_len, device, num_iters=20):
    # Warmup
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    for _ in range(5):
        _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iters):
        _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time_ms = ((end_time - start_time) / num_iters) * 1000
    return avg_time_ms

def measure_generation(model, prompt_len, gen_len, device, use_cache=False):
    # Warmup
    idx = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)
    generate(model, idx, max_new_tokens=2, temperature=1.0, use_cache=use_cache)
    
    start_time = time.time()
    out = generate(model, idx, max_new_tokens=gen_len, temperature=1.0, use_cache=use_cache)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    tokens_per_sec = gen_len / total_time
    return tokens_per_sec

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    # Use standard 1B parameters scale for benchmarking, or tiny scale for testing
    config = ModelConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=12,
        max_seq_len=1024
    )
    
    model = TransformerModel(config).to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f} M")
    
    # 1. Baseline Forward Pass
    print("\n--- Baseline Forward Pass ---")
    fwd_ms = measure_forward_pass(model, batch_size=4, seq_len=512, device=device)
    print(f"Forward Pass (B=4, T=512): {fwd_ms:.2f} ms")
    
    # 2. Baseline Generation (No KV Cache)
    print("\n--- Baseline Generation (No KV Cache) ---")
    tok_sec = measure_generation(model, prompt_len=10, gen_len=50, device=device, use_cache=False)
    print(f"Generation speed: {tok_sec:.2f} tokens/sec")

    # 3. Generation (With KV Cache)
    print("\n--- Generation (With KV Cache) ---")
    tok_sec_cache = measure_generation(model, prompt_len=10, gen_len=50, device=device, use_cache=True)
    print(f"Generation speed (KV Cache): {tok_sec_cache:.2f} tokens/sec")
    print(f"Speedup: {tok_sec_cache / tok_sec:.2f}x")
    
    # 4. Flash Attention Generation
    print("\n--- FlashAttention Generation (With KV Cache) ---")
    model.eval()
    # Switch backend to flash
    for layer in model.layers:
        from src.model.attention import FlashAttentionBackend
        layer.attn.backend = FlashAttentionBackend(model.config.dropout).to(device)
    
    tok_sec_flash = measure_generation(model, prompt_len=10, gen_len=50, device=device, use_cache=True)
    print(f"Generation speed (Flash + KV): {tok_sec_flash:.2f} tokens/sec")
    
    # 5. Compiled Forward Pass
    # PyTorch compilation works best on CUDA. CPU compilation can sometimes be slower.
    print("\n--- Compiled Forward Pass ---")
    try:
        print("Compiling model (this might take a moment)...")
        compiled_model = torch.compile(model)
        
        # compilation actually happens on first forward pass
        compile_start = time.time()
        x = torch.randint(0, model.config.vocab_size, (4, 512), device=device)
        _ = compiled_model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        compile_end = time.time()
        print(f"Compilation overhead: {compile_end - compile_start:.2f} s")
        
        comp_fwd_ms = measure_forward_pass(compiled_model, batch_size=4, seq_len=512, device=device)
        print(f"Compiled Forward Pass (B=4, T=512): {comp_fwd_ms:.2f} ms")
        print(f"Speedup: {fwd_ms / comp_fwd_ms:.2f}x")
    except Exception as e:
        print(f"Compilation failed or skipped: {e}")

if __name__ == '__main__':
    # Set PyTorch to utilize TensorCores if available
    torch.set_float32_matmul_precision('high')
    with torch.no_grad():
        main()
