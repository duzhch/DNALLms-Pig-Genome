#!/usr/bin/env python
"""
FASTA to TXT Converter for Pig Genome

This script converts pig genome FASTA files to a single text file,
removing non-ATCG characters and concatenating all chromosomes.

Usage:
    python fasta2txt.py --input_dir /path/to/fasta/files --output_file /path/to/output.txt
"""

import os
import re
import argparse
from Bio import SeqIO

# Parse command line arguments
def parse_args():
    """Parse command line arguments for the FASTA to TXT converter.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Convert pig genome FASTA files to a single text file')
    parser.add_argument('--input_dir', type=str, default='./data/fasta',
                        help='Directory containing FASTA files')
    parser.add_argument('--output_file', type=str, default='./data/processed/pig_genome.txt',
                        help='Output text file path')
    parser.add_argument('--genome_build', type=str, default='Sus_scrofa.Sscrofa11.1.dna.primary_assembly',
                        help='Genome build prefix for FASTA files')
    return parser.parse_args()

# Define main function
def main():
    """Main function to convert FASTA files to a single text file."""
    # Get command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Define chromosome order: chromosomes 1-18, MT, X, Y
    chromosomes = [str(i) for i in range(1, 19)] + ["MT", "X", "Y"]

    # Generate FASTA file paths list
    fasta_files = [f"{args.genome_build}.{chr}.fa" for chr in chromosomes]
    
    # Open output file in write mode
    with open(args.output_file, "w") as out_f:
        # Iterate through each FASTA file
        for fa_file in fasta_files:
            fa_path = os.path.join(args.input_dir, fa_file)
            
            # Check if file exists
            if not os.path.exists(fa_path):
                print(f"Warning: {fa_path} does not exist")
                continue
            
            # Parse FASTA file
            records = list(SeqIO.parse(fa_path, "fasta"))
            
            # Check record count (typically each FASTA file corresponds to one chromosome)
            if len(records) > 1:
                print(f"Warning: {fa_file} has {len(records)} records")
            
            # Process each record
            for record in records:
                # Extract sequence and convert to uppercase
                sequence = str(record.seq).upper()
                
                # Use regex to remove non-ATCG characters
                cleaned_sequence = re.sub(r'[^ATCG]', '', sequence)
                
                # Write cleaned sequence to output file
                out_f.write(cleaned_sequence)
    
    print(f"Processing completed. Output file: {args.output_file}")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()