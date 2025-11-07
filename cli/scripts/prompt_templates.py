"""
Prompt Templates for Different Base Models

Each model family has its own preferred instruction format.
Using the correct template improves model performance.
"""

TEMPLATES = {
    # Alpaca format (default, works with most models)
    "alpaca": {
        "system": "Below is an instruction that describes a task{context}. Write a response that appropriately completes the request.\n\n",
        "instruction": "### Instruction:\n{instruction}",
        "input": "\n\n### {input_label}:\n{input}",
        "response": "\n\n### Response:\n{response}",
        "inference": "### Instruction:\n{instruction}{context}\n\n### Response:\n",
    },
    
    # ChatML format (Qwen, Yi, Mistral)
    "chatml": {
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "instruction": "<|im_start|>user\n{instruction}",
        "input": "\n\n{input_label}:\n{input}",
        "response": "<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
        "inference": "<|im_start|>user\n{instruction}{context}<|im_end|>\n<|im_start|>assistant\n",
    },
    
    # Llama format (Llama 2, Llama 3, Code Llama)
    "llama": {
        "system": "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n",
        "instruction": "{instruction}",
        "input": "\n\n{input_label}:\n{input}",
        "response": " [/INST] {response} ",
        "inference": "[INST] {instruction}{context} [/INST] ",
    },
    
    # Mistral format
    "mistral": {
        "system": "",  # Mistral doesn't use system prompt in the same way
        "instruction": "[INST] {instruction}",
        "input": "\n\n{input_label}:\n{input}",
        "response": " [/INST] {response}</s>",
        "inference": "[INST] {instruction}{context} [/INST] ",
    },
    
    # Phi format (Phi-2, Phi-3, Phi-3.5)
    "phi": {
        "system": "System: {system}\n\n",
        "instruction": "User: {instruction}",
        "input": "\n\n{input_label}:\n{input}",
        "response": "\nAssistant: {response}",
        "inference": "User: {instruction}{context}\nAssistant: ",
    },
    
    # DeepSeek format
    "deepseek": {
        "system": "System: {system}\n\n",
        "instruction": "User: {instruction}",
        "input": "\n\n{input_label}:\n{input}",
        "response": "\n\nAssistant: {response}",
        "inference": "User: {instruction}{context}\n\nAssistant: ",
    },
    
    # Gemma format
    "gemma": {
        "system": "<start_of_turn>system\n{system}<end_of_turn>\n",
        "instruction": "<start_of_turn>user\n{instruction}",
        "input": "\n\n{input_label}:\n{input}",
        "response": "<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>",
        "inference": "<start_of_turn>user\n{instruction}{context}<end_of_turn>\n<start_of_turn>model\n",
    },
}


def get_template(template_name: str = "alpaca") -> dict:
    """Get prompt template by name"""
    if template_name not in TEMPLATES:
        print(f"[WARN] Template '{template_name}' not found, using 'alpaca'")
        return TEMPLATES["alpaca"]
    return TEMPLATES[template_name]


def format_training_example(
    instruction: str,
    response: str,
    input_text: str = None,
    input_label: str = "Input",
    template_name: str = "alpaca",
    system: str = None,
) -> str:
    """Format a training example using specified template"""
    template = get_template(template_name)
    
    # Start with instruction
    text = template["instruction"].format(instruction=instruction)
    
    # Add input/context if present
    if input_text:
        context = template["input"].format(
            input_label=input_label,
            input=input_text
        )
        text += context
    
    # Add response
    text += template["response"].format(response=response)
    
    # Add system prompt if present and template supports it
    if system and template.get("system"):
        text = template["system"].format(system=system) + text
    
    return text


def format_inference_prompt(
    instruction: str,
    input_text: str = None,
    input_label: str = "Input",
    template_name: str = "alpaca",
    system: str = None,
) -> str:
    """Format an inference prompt (without response)"""
    template = get_template(template_name)
    
    # Build context
    context = ""
    if input_text:
        context = template["input"].format(
            input_label=input_label,
            input=input_text
        )
    
    # Format prompt
    prompt = template["inference"].format(
        instruction=instruction,
        context=context
    )
    
    # Add system if present
    if system and template.get("system"):
        prompt = template["system"].format(system=system) + prompt
    
    return prompt


# Model-specific template recommendations
MODEL_TEMPLATES = {
    "qwen": "chatml",
    "qwen2": "chatml",
    "qwen3": "chatml",
    "yi": "chatml",
    "mistral": "mistral",
    "mixtral": "mistral",
    "llama": "llama",
    "llama-2": "llama",
    "llama-3": "llama",
    "codellama": "llama",
    "phi-2": "phi",
    "phi-3": "phi",
    "phi-3.5": "phi",
    "deepseek": "deepseek",
    "deepseek-coder": "deepseek",
    "gemma": "gemma",
    "gemma-2": "gemma",
}


def get_recommended_template(model_name: str) -> str:
    """Get recommended template based on model name"""
    model_lower = model_name.lower()
    
    for model_key, template in MODEL_TEMPLATES.items():
        if model_key in model_lower:
            return template
    
    # Default to alpaca (universal)
    return "alpaca"


if __name__ == "__main__":
    # Test templates
    instruction = "Find all persons named John"
    response = "MATCH (p:Person {name: 'John'}) RETURN p"
    schema = "Node properties:\n- Person (name: STRING)"
    
    print("=" * 60)
    print("Template Comparison")
    print("=" * 60)
    
    for name in ["alpaca", "chatml", "llama", "phi", "deepseek"]:
        print(f"\n{name.upper()} Format:")
        print("-" * 60)
        formatted = format_training_example(
            instruction, response, schema, "Schema", name
        )
        print(formatted[:300] + "..." if len(formatted) > 300 else formatted)

