import argparse
import os

import pandas as pd
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Generate protein sequences using ProtGPT2 model.')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of the generated sequences.')
    parser.add_argument('--top_k', type=int, default=950, help='Top-k sampling for generation.')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty for generation.')
    parser.add_argument('--num_return_sequences', type=int, default=10, help='Number of sequences to generate.')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
    sequences = protgpt2("<|endoftext|>",
                        max_length=args.max_length,
                        do_sample=True, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        num_return_sequences=args.num_return_sequences,
                        eos_token_id=args.eos_token_id)
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(sequences)
    
    work_dir = os.environ.get("WORK_DIR")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        print(f"Directory '{work_dir}' created.")
    
    data_path = os.path.join(work_dir, "sequences.csv")
    df.to_csv(data_path, index=False)

    print(f"Generated sequences saved to {data_path}")

    # Print the generated sequences
    for i, seq in enumerate(sequences):
        print(f"Sequence {i+1}: {seq['generated_text']}")

    
