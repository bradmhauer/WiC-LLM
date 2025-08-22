from tqdm import tqdm # To give us a progress bar.
from file_utils import read_data_owic, read_gold_owic
#from wic_ollama import llm_for_wic
from wic_hf import llm_for_wic

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Command line argument parser for model and data files.")

    parser.add_argument('--qwen_think', action='store_true', default=False,
                        help='Enable Qwen think mode (default: False)')

    parser.add_argument('--dev_data', type=str, default='owic/dev/dev.data.txt',
                        help='Path to the data file (default: owic/dev/dev.data.txt)')

    parser.add_argument('--dev_gold', type=str, default='owic/dev/dev.gold.txt',
                        help='Path to the gold file (default: owic/dev/dev.gold.txt)')

    parser.add_argument('--output', type=str, default='zeroshot_owic_dev.tsv',
                        help='Output file path (default: zeroshot_owic_dev.tsv)')

    args = parser.parse_args()
    return args


args = parse_arguments()
print("Parsed arguments:")
print(f"Qwen Think: {args.qwen_think}")
print(f"Dev Data: {args.dev_data}")
print(f"Dev Gold: {args.dev_gold}")
print(f"Output: {args.output}")
print()

print(f'Reading data from {args.dev_data}.')
df = read_data_owic(args.dev_data)

print(f'Computing {len(df)} answers with Qwen3.')
tqdm.pandas(desc="Processing rows")
df['answer'] = df.progress_apply(
    lambda x: llm_for_wic(
        x['lemma'], x['sentence1'], x['sentence2'],
        no_think = bool(1-args.qwen_think) # Iff qwen_think is true, no_think is False.
        ),
    axis=1,
    )
print('DONE!\n')

df['gold'] = read_gold_owic('owic/dev/dev.gold.txt')

df['eval'] = df['gold'] == df['answer']

print(df.head(10),'\n')

df.to_csv(args.output, sep='\t')
print(f'Output saved to {args.output}')

print()
print('Correct answers:', df['eval'].sum(), sep='\t')
print('Total instances:', len(df), sep='\t')
print('Result:         ', round(100 * df['eval'].mean(), 1), sep='\t')
