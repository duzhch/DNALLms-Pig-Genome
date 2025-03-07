import subprocess
import os
import json
import math
import torch
from typing import List, Union, Optional
import sentencepiece as spm
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

# 导入Mamba相关模块
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mamba_lm import MambaLMHeadModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
print(f"可用 GPU 数量: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置环境变量（保持不变）
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

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

# 路径和超参数配置（根据实际情况调整）
model_path = '/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/BPE/bpe_sentence/Pig_sen_bpe.model'
data_path = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/fasta_data/window1000_pig_genome.txt"
config_path = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/config.json"
run_path = "pig_mamba_run_test"
max_length = 1000  # 
batch_size =  256  # 总 batch size
learning_rate = 8e-5
num_train_epochs = 5  # 直接指定 epoch 数
warmup_ratio = 0.05
per_device_batch = 128
# 初始化分词器（保持不变）
tokenizer = SentencePieceTokenizer(model_path=model_path)

# 加载并转换模型配置
with open(config_path, "r") as f:
    config_dict = json.load(f)

# 将DNAmamba风格的配置转换为Mamba配置
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
    # 添加其他Mamba特定参数
)

# 初始化Mamba模型
model = MambaLMHeadModel(mamba_config).to(device)

# 数据集处理（保持与DNAmamba相同的处理流程）
raw_dataset = load_dataset('text', data_files=data_path)
dataset = raw_dataset["train"].train_test_split(test_size=0.1, shuffle=True)

def tokenize_function(examples):
    return tokenizer(examples['text'], 
                   truncation=True, 
                   padding='max_length', 
                   max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, 
                                batched=True, 
                                remove_columns=['text'], 
                                num_proc=15)

# 数据收集器（保持MLM设置）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 训练参数配置（优化适配Mamba）
training_args = TrainingArguments(
    output_dir=run_path,
    per_device_train_batch_size=per_device_batch,
    learning_rate=learning_rate,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    warmup_ratio=warmup_ratio,
    bf16=True,
    logging_steps=500,
    save_steps=2000,
    evaluation_strategy="steps",
    eval_steps=2000,
    num_train_epochs=num_train_epochs,
    dataloader_num_workers=10,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=2,
)

# 初始化Trainer（保持与DNAmamba相同接口）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# 训练流程（保持不变）
checkpoint_dir = get_last_checkpoint(run_path)
if checkpoint_dir:
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    trainer.train()

# 保存最终模型和分词器
trainer.save_model("mamba_dna_final")
tokenizer.save_pretrained("mamba_dna_final")

# 评估模型
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")