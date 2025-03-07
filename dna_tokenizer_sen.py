import sentencepiece as spm
from tqdm import tqdm
import mmap
import os

def fasta_window_generator(file_path, window_size=1000):
    """流式生成FASTA窗口的生成器"""
    current_seq = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):  # 跳过FASTA标题行
                if current_seq:
                    full_seq = ''.join(current_seq)
                    # 流式生成窗口
                    for i in range(0, len(full_seq), window_size):
                        yield full_seq[i:i+window_size]
                    current_seq = []
            else:
                current_seq.append(line)
        # 处理最后一个序列
        if current_seq:
            full_seq = ''.join(current_seq)
            for i in range(0, len(full_seq), window_size):
                yield full_seq[i:i+window_size]

def train_sentencepiece(file_path, model_prefix="Pig_sen_bpe", vocab_size=4096, window_size=1000):
    """使用SentencePiece训练模型并生成tokenizer"""
    # 生成一个临时的文本文件，用于SentencePiece训练
    temp_file = 'temp_train.txt'

    with open(temp_file, 'w') as temp_f:
        for window in tqdm(fasta_window_generator(file_path, window_size), desc="Generating training data"):
            temp_f.write(window + '\n')
    
    # 训练SentencePiece模型
    spm.SentencePieceTrainer.train(
        f'--input={temp_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe'
    )

    # 删除临时文件
    os.remove(temp_file)

def load_sentencepiece_tokenizer(model_prefix="Pig_sen_bpe"):
    """加载SentencePiece模型并返回Tokenizer对象"""
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

if __name__ == "__main__":
    file_path = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/fasta_data/pig_genome.txt"
    
    # 训练SentencePiece模型
    train_sentencepiece(file_path)
    
    # 加载训练好的SentencePiece模型
    tokenizer = load_sentencepiece_tokenizer()
    
    
    # 测试tokenizer
    test_seq = "TGGCGTGAACCCGGGATCGGG"
    print("Tokenized sequence:", tokenizer.encode(test_seq))  # SentencePiece使用encode方法
    print("Decoded sequence:", tokenizer.decode(tokenizer.encode(test_seq)))  # 解码回原始序列
