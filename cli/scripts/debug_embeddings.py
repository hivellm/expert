import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('F:/Node/hivellm/expert/models/Qwen3-0.6B', dtype=torch.bfloat16, device_map="cuda")
print('Embeddings shape:', model.model.embed_tokens.weight.shape)
print('Vocab size:', model.config.vocab_size)
print('Hidden size:', model.config.hidden_size)

# Test access
token_id = 14990
print(f'Token {token_id} embedding shape:', model.model.embed_tokens.weight[token_id].shape)
