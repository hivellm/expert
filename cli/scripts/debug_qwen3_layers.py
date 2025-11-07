#!/usr/bin/env python3
"""
Debug Qwen3 layer outputs - Python reference implementation
Extracts intermediate tensors for comparison with Rust
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_qwen3_forward(model_path: str, prompt: str, output_file: str):
    """Run forward pass and save all intermediate outputs"""
    
    print("="*60)
    print("Qwen3 Layer-by-Layer Debug (Python)")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Model loaded: {model.config.model_type}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Head dim: {model.config.head_dim}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Tokens: {len(input_ids[0])}")
    
    # Forward pass with hooks to capture intermediates
    debug_outputs = {
        "config": {
            "model_type": model.config.model_type,
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "head_dim": model.config.head_dim,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
        },
        "input": {
            "prompt": prompt,
            "input_ids": input_ids[0].cpu().tolist(),
            "num_tokens": len(input_ids[0]),
        },
        "embeddings": {},
        "layers": [],
        "final": {},
    }
    
    with torch.no_grad():
        # Use model.forward() to get outputs properly
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden_states_list = outputs.hidden_states
        
        # 1. Embeddings (first hidden state)
        if hidden_states_list:
            embeddings = hidden_states_list[0]
            debug_outputs["embeddings"] = {
                "shape": list(embeddings.shape),
                "mean": embeddings.mean().item(),
                "std": embeddings.std().item(),
                "min": embeddings.min().item(),
                "max": embeddings.max().item(),
                "first_5": embeddings[0, 0, :5].cpu().tolist(),
            }
            
            print(f"\nEmbeddings shape: {embeddings.shape}")
            print(f"  Mean: {embeddings.mean():.6f}")
            print(f"  Std:  {embeddings.std():.6f}")
        
        # 2. Process each layer output
        for idx in range(len(hidden_states_list) - 1):
            print(f"\nLayer {idx}:")
            
            layer_input = hidden_states_list[idx]
            layer_output = hidden_states_list[idx + 1]
            
            layer_debug = {
                "layer_idx": idx,
                "input": {
                    "shape": list(layer_input.shape),
                    "mean": layer_input.mean().item(),
                    "std": layer_input.std().item(),
                },
                "output": {
                    "shape": list(layer_output.shape),
                    "mean": layer_output.mean().item(),
                    "std": layer_output.std().item(),
                    "min": layer_output.min().item(),
                    "max": layer_output.max().item(),
                },
            }
            
            print(f"  Input - Mean: {layer_input.mean():.6f}, Std: {layer_input.std():.6f}")
            print(f"  Output - Mean: {layer_output.mean():.6f}, Std: {layer_output.std():.6f}")
            
            debug_outputs["layers"].append(layer_debug)
        
        # 3. Final hidden state (before norm)
        final_hidden = hidden_states_list[-1] if hidden_states_list else logits
        debug_outputs["final"]["hidden"] = {
            "shape": list(final_hidden.shape),
            "mean": final_hidden.mean().item(),
            "std": final_hidden.std().item(),
        }
        
        print(f"\nFinal hidden state:")
        print(f"  Mean: {final_hidden.mean():.6f}")
        print(f"  Std:  {final_hidden.std():.6f}")
        
        # 4. Logits (last token)
        logits = logits[:, -1, :]  # Last token
        
        debug_outputs["final"]["logits"] = {
            "shape": list(logits.shape),
            "mean": logits.mean().item(),
            "std": logits.std().item(),
            "min": logits.min().item(),
            "max": logits.max().item(),
            "argmax": logits.argmax(dim=-1).item(),
            "top_5_tokens": logits.topk(5).indices[0].cpu().tolist(),
            "top_5_probs": torch.softmax(logits, dim=-1).topk(5).values[0].cpu().tolist(),
        }
        
        print(f"\nLogits:")
        print(f"  Shape: {logits.shape}")
        print(f"  Mean: {logits.mean():.6f}")
        print(f"  Argmax token: {logits.argmax(dim=-1).item()}")
        print(f"  Top 5 tokens: {logits.topk(5).indices[0].tolist()}")
        
        # Decode
        next_token = logits.argmax(dim=-1)
        decoded = tokenizer.decode([next_token.item()])
        debug_outputs["final"]["predicted_token"] = next_token.item()
        debug_outputs["final"]["predicted_text"] = decoded
        
        print(f"  Predicted: '{decoded}'")
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(debug_outputs, f, indent=2)
    
    print(f"\n[OK] Debug outputs saved to: {output_path}")
    print(f"   {output_path.stat().st_size} bytes")
    
    return debug_outputs

if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "debug_python.json"
    
    debug_qwen3_forward(model_path, prompt, output_file)

