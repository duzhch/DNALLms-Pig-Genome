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
import os


file_path = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/fasta_data/pig_genome.txt"

with open(file_path, 'r') as file:
    text = file.read()
window_size = 1000
windows = [text[i:i+window_size] for i in range(0, len(text), window_size)]

# 保存分割后的窗口数据到新的文本文件
window_file_path = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/fasta_data/window1000_pig_genome.txt"
with open(window_file_path, 'w') as window_file:
    for window in windows:
        window_file.write(window + '\n')


tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)  # use_regex=False,空格当成一般字符串
trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<|endoftext|>"])  # 2^12 4096 dnabert-2 

tokenizer.train([window_file_path], trainer=trainer)

tokenizer.save("dna_bpe_pig.json")

new_tokenizer = Tokenizer.from_file("dna_bpe_pig.json")

# 在 Hugging Face Transformers 中使用此分词器
from transformers import GPT2TokenizerFast
dna_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)
dna_tokenizer.save_pretrained("dna_bpe_pig")

# 上传你的模型到Hugging Face Hub
# dna_tokenizer.push_to_hub("dna_bpe_dict_1g", organization="dnagpt", use_auth_token="hf_*****")  # push to huggingface

tokenizer_new = AutoTokenizer.from_pretrained('dna_bpe_pig')
print(tokenizer_new.tokenize("TGGCGTGAACCCGGGATCGGG"))
