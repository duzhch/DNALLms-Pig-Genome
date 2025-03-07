import subprocess
import os
import json
import math
import torch
from typing import List, Union, Optional
import sentencepiece as spm
from transformers import AutoConfig, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizer
from datasets import load_dataset

# Set environment variables if needed
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# 自定义 SentencePiece 分词器
class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(self, model_path: str, **kwargs):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self._vocab_size = self.sp.get_piece_size()
        super().__init__(**kwargs)
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.pad_token = self.eos_token
        self.pad_token_id = self.sp.piece_to_id(self.pad_token) if self.sp.piece_to_id(self.pad_token) < self._vocab_size else 0
        self.eos_token_id = self.sp.piece_to_id(self.eos_token) if self.sp.piece_to_id(self.eos_token) < self._vocab_size else 2
        self.unk_token_id = self.sp.piece_to_id(self.unk_token) if self.sp.piece_to_id(self.unk_token) < self._vocab_size else 1
        self.bos_token_id = self.sp.piece_to_id(self.bos_token) if self.sp.piece_to_id(self.bos_token) < self._vocab_size else 3

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

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = False, **kwargs) -> List[int]:
        if isinstance(text, str):
            return self.sp.encode(text, out_type=int)
        return [self.sp.encode(t, out_type=int) for t in text]

    def decode(self, ids: Union[int, List[int]], skip_special_tokens: bool = False, **kwargs) -> str:
        return self.sp.decode(ids)

    def __call__(self, text, padding: bool = True, truncation: bool = True, max_length: int = 512, return_tensors: str = "pt", **kwargs):
        if isinstance(text, str):
            tokens = self.encode(text)
        else:
            tokens = self.encode(text)
        if truncation:
            tokens = [t[:max_length] for t in tokens] if isinstance(text, list) else tokens[:max_length]
        if padding:
            max_len = min(max_length, max(len(t) for t in tokens) if isinstance(text, list) else len(tokens))
            tokens = [t + [self.pad_token_id] * (max_len - len(t)) for t in tokens] if isinstance(text, list) else tokens + [self.pad_token_id] * (max_len - len(tokens))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(tokens if isinstance(text, list) else [tokens])}
        return {"input_ids": tokens if isinstance(text, list) else [tokens]}

# Initialize argument parser
parser = argparse.ArgumentParser(description='GPT-2 Model Training Configuration')
parser.add_argument('--model_path', type=str, required=True, help='Path to SentencePiece model')
parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
parser.add_argument('--config_path', type=str, default='config.json', help='Path to model config file')
parser.add_argument('--run_path', type=str, default='gpt2_run', help='Output directory for training results')
parser.add_argument('--output_dir', type=str, default='pig_gpt2_models', help='Final model output directory')
parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
parser.add_argument('--train_epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Training batch size')
args = parser.parse_args()

# 初始化分词器
tokenizer = SentencePieceTokenizer(model_path=model_path)
tokenizer.pad_token = tokenizer.eos_token

# 保存词汇表和合并文件（如果尚未保存）
vocab = {tokenizer.sp.id_to_piece(i): i for i in range(tokenizer.vocab_size)}
vocab_file = os.path.join(args.run_path, "vocab.json")
merge_file = os.path.join(args.run_path, "merges.txt")
with open(vocab_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
with open(merge_file, "w", encoding="utf-8") as f:
    f.write("# Empty merges file\n")

# 加载模型配置
config = AutoConfig.from_pretrained(
    args.config_path,
    vocab_size=len(tokenizer),
    n_ctx=max_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)

# 加载数据集
raw_dataset = load_dataset('text', data_files=data_path)
dataset = raw_dataset["train"].train_test_split(test_size=0.1, shuffle=True)

# 对数据集进行分词
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'], num_proc=15)

# 数据收集器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir=run_path,
    overwrite_output_dir=False,  # 设置为 False 以避免覆盖检查点
    num_train_epochs=train_epochs,
    per_device_train_batch_size=batch_size,
    save_steps=2000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
)

# 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# 从检查点恢复训练
if os.path.exists(run_path) and any(os.path.isdir(os.path.join(run_path, d)) for d in os.listdir(run_path)):
    print(f"从 {run_path} 中的检查点恢复训练")
    trainer.train(resume_from_checkpoint=True)
else:
    print(f"在 {run_path} 中未找到检查点，从头开始训练")
    trainer.train()

# 保存最终模型
trainer.save_model(args.output_dir)

# 评估模型
eval_results = trainer.evaluate()
print(f"困惑度: {math.exp(eval_results['eval_loss']):.2f}")