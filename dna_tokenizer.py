#!/usr/bin/env python
"""
DNA Tokenizer for Pig Genome

This script creates a BPE tokenizer for DNA sequences from a processed genome text file.
It splits the genome into windows of fixed size and trains a BPE tokenizer on these sequences.

Usage:
    python dna_tokenizer.py --input_file /path/to/genome.txt --output_dir /path/to/output --window_size 1000
"""

import os
import argparse
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import AutoTokenizer

# Parse command line arguments
def parse_args():
    """Parse command line arguments for the DNA tokenizer.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Create a BPE tokenizer for DNA sequences')
    parser.add_argument('--input_file', type=str, default='./data/processed/pig_genome.txt',
                        help='Input genome text file path')
    parser.add_argument('--output_dir', type=str, default='./models/tokenizer',
                        help='Output directory for tokenizer files')
    parser.add_argument('--window_size', type=int, default=1000,
                        help='Window size for splitting genome sequences')
    parser.add_argument('--vocab_size', type=int, default=4096,
                        help='Vocabulary size for the tokenizer')
    return parser.parse_args()

def main():
    """Main function to create and train a DNA tokenizer."""
    # Get command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the genome text file
    with open(args.input_file, 'r') as file:
        text = file.read()
    
    # Split the genome into windows of fixed size
    window_size = args.window_size
    windows = [text[i:i+window_size] for i in range(0, len(text), window_size)]
    
    # Save the windowed data to a new text file
    window_file_path = os.path.join(os.path.dirname(args.input_file), f"window{window_size}_pig_genome.txt")
    with open(window_file_path, 'w') as window_file:
        for window in windows:
            window_file.write(window + '\n')

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)  # Use regex=False to treat spaces as normal characters
trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=[""])  # Default vocab size: 4096 (2^12) like DNABERT-2

tokenizer.train([window_file_path], trainer=trainer)

# Save tokenizer to output directory
tokenizer_json_path = os.path.join(args.output_dir, "dna_bpe_pig.json")
tokenizer.save(tokenizer_json_path)

new_tokenizer = Tokenizer.from_file(tokenizer_json_path)

# Use the tokenizer with Hugging Face Transformers
from transformers import GPT2TokenizerFast
dna_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)
dna_tokenizer.save_pretrained(os.path.join(args.output_dir, "dna_bpe_pig"))

# 上传你的模型到Hugging Face Hub
# dna_tokenizer.push_to_hub("dna_bpe_dict_1g", organization="dnagpt", use_auth_token="hf_*****")  # push to huggingface

# Optionally upload to Hugging Face Hub (commented out by default)
# dna_tokenizer.push_to_hub("dna_bpe_pig", use_auth_token="YOUR_HF_TOKEN")

if __name__ == "__main__":
    main()
