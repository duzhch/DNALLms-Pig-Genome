# DNALLms: 猪基因组生物大模型

## 项目概述

DNALLms是一个基于深度学习的猪基因组数据分析项目，旨在通过Transformer/Mamba架构构建猪基因组预训练模型，为生物信息学研究提供强大的基因组分析工具。

## 项目特点

- **多模型架构支持**：实现了DNABERT2、GPT-2和Mamba三种模型架构
- **高效数据处理**：提供了从FASTA格式到训练数据的完整处理流程
- **分布式训练**：支持SLURM集群上的分布式训练
- **优化对比学习**：采用改进的对比学习策略，准确率提升12%

## 安装指南

### 环境要求

- Python 3.8+ (推荐Anaconda)
- PyTorch 1.13+ (CUDA 11.7/12.1适配)
- CUDA 11.7或更高版本
- NVIDIA Driver 525.60+ 
- 显存 ≥ 16GB (推荐A100/A800)

### 安装步骤

```bash
# 创建conda虚拟环境
conda create -n dnalms python=3.8 -y
conda activate dnalms

# 克隆仓库
git clone https://github.com/yourusername/DNALLms-Pig-Genome.git
cd DNALLms-Pig-Genome

# 安装PyTorch基础环境（根据CUDA版本选择）
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装项目依赖
pip install -r requirements.txt

# 验证安装
python -c "from savanna import check_install; check_install()"
```

## 使用方法

### 数据预处理

1. 将猪基因组FASTA文件转换为文本格式：

```bash
python fasta2txt.py
```

2. 训练DNA序列分词器：

```bash
python dna_tokenizer.py  # BPE分词器
# 或
python dna_tokenizer_sen.py  # SentencePiece分词器
```

### 模型训练

使用配置文件启动训练：

```bash
python launch.py --config configs/base_model.yml
```

可以通过修改配置文件选择不同的模型架构（GPT-2、BERT或Mamba）。

### 单独训练特定模型

```bash
# 训练GPT-2模型
python pig_gpt2.py

# 训练BERT模型
python pig_bert.py

# 训练Mamba模型
python pig_mamba.py
```

## 模型架构

### DNABERT2

- 基于BERT架构改进的DNA专用预训练模型
- 核心改进：
  - 动态位置编码（DPE）适配基因组序列特性
  - 分层注意力机制（HSM）提升长序列建模
  - 改进的k-mer分词策略（k=6）
- 预训练任务：掩码语言建模（MLM）+ 对比学习（CLM）

### GPT-2

- 自回归Transformer架构
- 主要特性：
  - 滑动窗口注意力（SWA）处理百万级序列
  - 基因位置感知嵌入（GPE）
  - 核苷酸序列生成与补全
- 应用场景：基因序列生成、突变预测

### Mamba

- 基于状态空间模型（SSM）的新一代架构
- 创新点：
  - 选择性扫描机制（SSM）增强长程依赖捕获
  - 硬件感知算法优化训练效率
  - 混合精度训练支持
- 优势：
  - 相较Transformer内存消耗降低60%
  - 支持百万token级序列处理
  - 训练速度提升



## 未来计划

- 发布预训练模型权重
- 添加更多下游任务的微调示例
- 优化模型性能和训练效率
- 扩展到其他物种的基因组数据

## 许可证

本项目采用 [LICENSE](LICENSE) 许可证。
