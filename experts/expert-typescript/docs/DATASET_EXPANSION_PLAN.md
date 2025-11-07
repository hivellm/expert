# Dataset Expansion Plan - Expert TypeScript

## Objective
Significantly expand the expert-typescript dataset from 20k to 100k+ examples, including official documentation and relevant datasets.

## Identified Datasets

### 1. Official TypeScript Documentation
- **Source**: https://www.typescriptlang.org/docs/
- **Content**: TypeScript Handbook, code examples, tutorials
- **Estimate**: ~5,000-10,000 examples (extracted from code + explanations)
- **Format**: Instruction-output pairs (question about concept → code example)

### 2. HuggingFace Datasets

#### bigcode/the-stack (TypeScript subset)
- **Dataset**: bigcode/the-stack
- **Filter**: Language = TypeScript
- **Size**: ~500GB+ of TypeScript code
- **License**: Permissive (BigCode OpenRAIL-M)
- **Usage**: Extract code examples with context
- **Estimate**: ~50,000-100,000 unique examples

#### ManyTypes4TypeScript
- **Dataset**: ManyTypes4TypeScript (via API)
- **Content**: 9+ million type annotations, 13,953 projects
- **Focus**: Type inference, type annotations
- **Estimate**: ~20,000 examples (representative sample)

#### WizardLM/WizardLM_evol_instruct_V2_196k
- **Dataset**: WizardLM_evol_instruct_V2_196k
- **Filter**: TypeScript-related examples
- **Content**: Evolved instructions, generated code
- **Estimate**: ~5,000-10,000 TypeScript examples

#### mhhmm/typescript-instruct-20k-v2c (current)
- **Dataset**: Already in use
- **Size**: 20,000 examples
- **Keep**: Yes, as base

### 3. Open Source Repositories (optional)
- **Source**: GitHub (popular TypeScript projects)
- **Examples**: VS Code, Angular, NestJS
- **License**: Check permissiveness
- **Estimate**: ~10,000-20,000 examples

## Integration Strategy

### Phase 1: Official Documentation (High Priority)
1. Extract examples from TypeScript Handbook
2. Create instruction-output pairs
3. Format in ChatML
4. Add to synthetic_fixes.jsonl

### Phase 2: The Stack TypeScript (High Priority)
1. Filter TypeScript code from the-stack
2. Extract functions/classes with context
3. Create instructions based on code
4. Validate TypeScript syntax
5. Remove duplicates

### Phase 3: ManyTypes4TypeScript (Medium Priority)
1. Access via API
2. Extract type inference examples
3. Create type-focused instruction-output pairs
4. Integrate into dataset

### Phase 4: WizardLM (Medium Priority)
1. Filter TypeScript examples
2. Validate quality
3. Integrate into dataset

## Dataset Format

### Current Format (mhhmm/typescript-instruct-20k-v2c)
```json
{
  "instruction": "Create a function that validates email addresses",
  "output": "function validateEmail(email: string): boolean { ... }"
}
```

### ChatML Format (for training)
```json
{
  "text": "<|system|>\nDialect: typescript\n<|end|>\n<|user|>\nCreate a function that validates email addresses\n<|end|>\n<|assistant|>\nfunction validateEmail(email: string): boolean { ... }\n<|end|>"
}
```

## Required Scripts

1. **extract_typescript_docs.py**: Extract official documentation
2. **integrate_the_stack.py**: Integrate the-stack TypeScript subset
3. **integrate_manytypes.py**: Integrate ManyTypes4TypeScript
4. **merge_datasets.py**: Combine all datasets
5. **validate_typescript.py**: Validate TypeScript syntax (tsc)

## Goals

- **Final Dataset**: 100,000+ examples
- **Coverage**: 
  - Basic concepts (20%)
  - Advanced types (25%)
  - Code patterns (25%)
  - Frameworks/APIs (15%)
  - Refactoring/explanation (15%)

## Next Steps

1. ✅ Create script structure
2. ⏳ Extract official documentation
3. ⏳ Integrate the-stack TypeScript
4. ⏳ Validate and deduplicate
5. ⏳ Train expanded model

