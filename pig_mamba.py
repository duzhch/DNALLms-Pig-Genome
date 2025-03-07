#!/usr/bin/env python
"""
Mamba Model Training for Pig Genome

This script implements a Mamba-based language model for pig genome sequences.
It uses the state space model architecture which is more efficient for long sequences.

Usage:
    python pig_mamba.py --model_path /path/to/tokenizer --data_path /path/to/data --output_dir /path/to/output
"""

import subprocess
import os
import json
import math
import torch
import argparse
from typing import List, Union, Optional
import sentencepiece as spm
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

# Import Mamba-related modules
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mamba_lm import MambaLMHeadModel

# Parse command line arguments
def parse_args():
    """Parse command line arguments for the Mamba model training.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a Mamba model on pig genome data')
    parser.add_argument('--model_path', type=str, default='./models/tokenizer/Pig_sen_bpe.model',
                        help='Path to the SentencePiece model file')
    parser.add_argument('--data_path', type=str, default='./data/processed/window1000_pig_genome.txt',
                        help='Path to the processed genome data file')
    parser.add_argument('--config_path', type=str, default='./configs/mamba_config.json',
                        help='Path to the model configuration file')
    parser.add_argument('--output_dir', type=str, default='./models/mamba',
                        help='Directory to save the trained model')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Total batch size for training')
    parser.add_argument('--per_device_batch', type=int, default=128,
                        help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=8e-5,
                        help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Warmup ratio for learning rate scheduler')
    return parser.parse_args()

# Setup CUDA environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
print(f"Available GPUs: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(self, model_path: str, **kwargs):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self._vocab_size = self.sp.get_piece_size()
        super().__init__(**kwargs)
        self.pad_token = "[PAD]"
        self.eos_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.pad_token_id = self.sp.piece_to_id(self.pad_token) if self.sp.piece_to_id(self.pad_token) < self._vocab_size else 0
        self.eos_token_id = self.sp.piece_to_id(self.eos_token) if self.sp.piece_to_id(self.eos_token) < self._vocab_size else 2
        self.unk_token_id = self.sp.piece_to_id(self.unk_token) if self.sp.piece_to_id(self.unk_token) < self._vocab_size else 1
        self.cls_token_id = self.sp.piece_to_id(self.cls_token) if self.sp.piece_to_id(self.cls_token) < self._vocab_size else 3
        self.mask_token_id = self.sp.piece_to_id(self.mask_token) if self.sp.piece_to_id(self.mask_token) < self._vocab_size else 4

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> dict:
        return {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp.id_to_piece(index)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True, **kwargs) -> List[int]:
        if isinstance(text, str):
            encoded = self.sp.encode(text, out_type=int)
            if add_special_tokens:
                return [self.cls_token_id] + encoded + [self.eos_token_id]
            return encoded
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in text]

    def decode(self, ids: Union[int, List[int]], skip_special_tokens: bool = True, **kwargs) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in {self.pad_token_id, self.cls_token_id, self.eos_token_id, self.mask_token_id}]
        return self.sp.decode(ids)

    def __call__(self, text, padding: bool = True, truncation: bool = True, max_length: int = 128, return_tensors: str = "pt", **kwargs):
        if isinstance(text, str):
            tokens = self.encode(text, add_special_tokens=True)
        else:
            tokens = self.encode(text, add_special_tokens=True)
        if truncation:
            tokens = [t[:max_length] for t in tokens] if isinstance(text, list) else tokens[:max_length]
        if padding:
            max_len = min(max_length, max(len(t) for t in tokens) if isinstance(text, list) else len(tokens))
            tokens = [t + [self.pad_token_id] * (max_len - len(t)) for t in tokens] if isinstance(text, list) else tokens + [self.pad_token_id] * (max_len - len(tokens))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(tokens if isinstance(text, list) else [tokens])}
        return {"input_ids": tokens if isinstance(text, list) else [tokens]}

class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(self, model_path: str, **kwargs):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self._vocab_size = self.sp.get_piece_size()
        super().__init__(**kwargs)
        self.pad_token = "[PAD]"
        self.eos_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.pad_token_id = self.sp.piece_to_id(self.pad_token) if self.sp.piece_to_id(self.pad_token) < self._vocab_size else 0
        self.eos_token_id = self.sp.piece_to_id(self.eos_token) if self.sp.piece_to_id(self.eos_token) < self._vocab_size else 2
        self.unk_token_id = self.sp.piece_to_id(self.unk_token) if self.sp.piece_to_id(self.unk_token) < self._vocab_size else 1
        self.cls_token_id = self.sp.piece_to_id(self.cls_token) if self.sp.piece_to_id(self.cls_token) < self._vocab_size else 3
        self.mask_token_id = self.sp.piece_to_id(self.mask_token) if self.sp.piece_to_id(self.mask_token) < self._vocab_size else 4

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> dict:
        return {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp.id_to_piece(index)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True, **kwargs) -> List[int]:
        if isinstance(text, str):
            encoded = self.sp.encode(text, out_type=int)
            if add_special_tokens:
                return [self.cls_token_id] + encoded + [self.eos_token_id]
            return encoded
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in text]

    def decode(self, ids: Union[int, List[int]], skip_special_tokens: bool = True, **kwargs) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in {self.pad_token_id, self.cls_token_id, self.eos_token_id, self.mask_token_id}]
        return self.sp.decode(ids)

    def __call__(self, text, padding: bool = True, truncation: bool = True, max_length: int = 128, return_tensors: str = "pt", **kwargs):
        if isinstance(text, str):
            tokens = self.encode(text, add_special_tokens=True)
        else:
            tokens = self.encode(text, add_special_tokens=True)
        if truncation:
            tokens = [t[:max_length] for t in tokens] if isinstance(text, list) else tokens[:max_length]
        if padding:
            max_len = min(max_length, max(len(t) for t in tokens) if isinstance(text, list) else len(tokens))
            tokens = [t + [self.pad_token_id] * (max_len - len(t)) for t in tokens] if isinstance(text, list) else tokens + [self.pad_token_id] * (max_len - len(tokens))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(tokens if isinstance(text, list) else [tokens])}
        return {"input_ids": tokens if isinstance(text, list) else [tokens]}

def main():
    # Get command line arguments
    args = parse_args()
    
    # Initialize tokenizer
    tokenizer = SentencePieceTokenizer(model_path=args.model_path)

# Load and convert model configuration
with open(args.config_path, "r") as f:
    config_dict = json.load(f)

# Convert DNA-specific configuration to Mamba configuration
mamba_config = MambaConfig(
    d_model=config_dict["hidden_size"],
    n_layer=config_dict["num_hidden_layers"],
    vocab_size=tokenizer.vocab_size,
    d_intermediate=config_dict.get("intermediate_size", 0),
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    pad_vocab_size_multiple=8,
    tie_embeddings=True,
    # Add other Mamba-specific parameters as needed
)

# Initialize Mamba model
model = MambaLMHeadModel(mamba_config).to(device)

# Dataset processing
raw_dataset = load_dataset('text', data_files=args.data_path)
dataset = raw_dataset["train"].train_test_split(test_size=0.1, shuffle=True)

def tokenize_function(examples):
    return tokenizer(examples['text'], 
                   truncation=True, 
                   padding='max_length', 
                   max_length=args.max_length)

tokenized_datasets = dataset.map(tokenize_function, 
                                batched=True, 
                                remove_columns=['text'], 
                                num_proc=15)

# Data collator (maintain MLM settings)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments configuration (optimized for Mamba)
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_batch,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    warmup_ratio=args.warmup_ratio,
    bf16=True,
    logging_steps=500,
    save_steps=2000,
    evaluation_strategy="steps",
    eval_steps=2000,
    num_train_epochs=args.num_train_epochs,
    dataloader_num_workers=10,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=2,
)

# Initialize Trainer (maintain same interface as DNAmamba)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# Training process (keep unchanged)
checkpoint_dir = get_last_checkpoint(args.output_dir)
if checkpoint_dir:
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    trainer.train()

# Save final model and tokenizer
final_model_path = os.path.join(args.output_dir, "mamba_dna_final")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Evaluate model
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    main()