import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('F:/Node/hivellm/expert/models/Qwen3-0.6B')
tokens = tokenizer.encode('hello', add_special_tokens=True)
print('Token IDs:', tokens)
for i, token_id in enumerate(tokens):
    token_text = tokenizer.decode([token_id])
    print(f'  {i}: {token_id} -> "{token_text}"')
