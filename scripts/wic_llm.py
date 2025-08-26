"""
Main driver script for WiC experiments.
Loads data and gold labels,
runs an LLM backend (Transformers or Ollama),
saves predictions and evaluation results.

Usage examples:
python wic_llm.py --framework transformers --model Qwen/Qwen3-1.7B
python wic_llm.py --framework ollama --model qwen3:1.7b-fp16
"""

import argparse
from tqdm import tqdm # To give us a progress bar.

from wic_utils import read_data_owic, read_gold_owic

try:
    import wic_transformers
except ImportError:
    wic_transformers = None

try:
    import wic_ollama
except ImportError:
    wic_ollama = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Command line argument parser for model and data files.")

    parser.add_argument('--framework', type=str, default='transformers',
                        help='The backend to use for LLM inference.')

    parser.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B',
                        help='The LLM to be used.')

    parser.add_argument('--seed', type=int, default=9999,
                        help='The random seed, for reproducibility.')
    
    parser.add_argument('--qwen_think', action='store_true', default=False,
                        help='Enable Qwen think mode (default: False)')

    parser.add_argument('--data', type=str, default='../data/dev/dev.data.txt',
                        help='Path to the data file (default: owic/dev/dev.data.txt)')

    parser.add_argument('--gold', type=str, default='../data/dev/dev.gold.txt',
                        help='Path to the gold file (default: owic/dev/dev.gold.txt)')

    parser.add_argument('--output', type=str, default='../results/results_owic_dev.tsv',
                        help='Output file path (default: zeroshot_owic_dev.tsv)')

    args = parser.parse_args()
    return args


args = parse_arguments()
print("Parsed arguments:")
print(f"Framework:  {args.framework}")
print(f"Model:      {args.model}")
print(f"Seed:       {args.seed}")
print(f"Qwen Think: {args.qwen_think}")
print(f"Dev Data:   {args.data}")
print(f"Dev Gold:   {args.gold}")
print(f"Output:     {args.output}")
print()


if args.framework == "transformers":
    if wic_transformers is None:
        raise ImportError("Transformers backend not available. Please install Transformers.")
    llm_for_wic = wic_transformers.llm_for_wic

elif args.framework == "ollama":
    if wic_ollama is None:
        raise ImportError("Ollama backend not available. Please install Ollama.")
    llm_for_wic = wic_ollama.llm_for_wic


print(f'Reading data from {args.data}.')
df = read_data_owic(args.data)

print(f'Computing {len(df)} answers with {args.model}.')
tqdm.pandas(desc="Processing rows")
df['answer'] = df.progress_apply(
    lambda instance: llm_for_wic(
        instance['lemma'],
        instance['sentence1'],
        instance['sentence2'],
        model = args.model,
        seed = args.seed,
        no_think = not args.qwen_think # Iff qwen_think is true, no_think is False.
        ),
    axis=1,
    )
print('DONE!\n')

df['gold'] = read_gold_owic(args.gold)

df['eval'] = df['gold'] == df['answer']

print(df.head(10),'\n')

df.to_csv(args.output, sep='\t')
print(f'Output saved to {args.output}')

print()
print('Correct answers:', df['eval'].sum(), sep='\t')
print('Total instances:', len(df), sep='\t')
print('Result:         ', round(100 * df['eval'].mean(), 1), sep='\t')
