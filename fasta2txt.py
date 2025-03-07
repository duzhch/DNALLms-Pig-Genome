import os
from Bio import SeqIO
import re

# 设置FASTA文件所在的文件夹路径
# fasta_dir = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/fasta_data"
fasta_dir = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/Piggenome"

# 定义染色体顺序：1-18号染色体，MT，X，Y
chromosomes = [str(i) for i in range(1, 19)] + ["MT", "X", "Y"]

# 生成FASTA文件路径列表
fasta_files = [f"Sus_scrofa.Sscrofa11.1.dna.primary_assembly.{chr}.fa" for chr in chromosomes]

# 设置输出文件路径
output_file = "/work/home/zyqgroup01/duanzhichao/GLM/PIGMODEL/fasta_data/pig_genome.txt"

# 打开输出文件，写入模式
with open(output_file, "w") as out_f:
    # 遍历每个FASTA文件
    for fa_file in fasta_files:
        fa_path = os.path.join(fasta_dir, fa_file)
        
        # 检查文件是否存在
        if not os.path.exists(fa_path):
            print(f"Warning: {fa_path} does not exist")
            continue
        
        # 解析FASTA文件
        records = list(SeqIO.parse(fa_path, "fasta"))
        
        # 检查记录数量（通常每个FASTA文件对应一个染色体，应只有1个记录）
        if len(records) > 1:
            print(f"Warning: {fa_file} has {len(records)} records")
        
        # 遍历每个记录
        for record in records:
            # 提取序列并转换为大写
            sequence = str(record.seq).upper()
            
            # 使用正则表达式去除非ATCG字符
            cleaned_sequence = re.sub(r'[^ATCG]', '', sequence)
            
            # 将清洗后的序列写入输出文件
            out_f.write(cleaned_sequence)

print(f"Pre_processing completed. Output file: {output_file}")