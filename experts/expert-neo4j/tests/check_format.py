from datasets import load_dataset

ds = load_dataset('neo4j/text2cypher-2025v1', split='train')
print('Columns:', ds.column_names)

ex = ds[0]
print('\n' + '='*60)
print('Example from dataset:')
print('='*60)
print(f'Question: {ex["question"]}')
print(f'\nCypher: {ex["cypher"]}')

if 'input' in ex and ex['input']:
    print(f'\nInput: {ex["input"]}')

print('\n' + '='*60)
print('Training format (auto-detected by expert_trainer.py):')
print('='*60)
print(f'### Instruction:\n{ex["question"]}\n\n### Response:\n{ex["cypher"]}')

print('\n' + '='*60)
print('The schema field was NOT used in training prompt!')
print('It was part of the dataset but NOT in the instruction text.')
print('='*60)

