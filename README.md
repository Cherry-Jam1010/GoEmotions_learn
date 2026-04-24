# GoEmotions 情绪分类分析

基于 GoEmotions 数据集的 27 类细粒度情绪分类研究，采用 TF-IDF 和 Word2Vec 特征提取，结合 K-Means 和层次聚类算法探索情绪语义空间，并将结果通过 t-SNE 可视化。

[简体中文报告](report.html) · [English Report](#)

---

## 目录

- [数据集](#数据集)
- [项目结构](#项目结构)
- [情绪分类体系](#情绪分类体系)
- [快速开始](#快速开始)
- [技术流程](#技术流程)
- [主要发现](#主要发现)
- [输出文件](#输出文件)
- [引用](#引用)

---

## 数据集

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| 训练集 | 43,410 | 模型训练 |
| 验证集 | 5,426 | 超参数调优 |
| 测试集 | 5,427 | 最终评估 |

- 共 **58,009** 条 Reddit 评论，由人工标注至 **27 类情绪**
- 原始标注可有多标签，筛选至少 2 名标注者一致的数据
- 数据来源：[GoEmotions (Google Research, ACL 2020)](https://arxiv.org/abs/2005.00547)

---

## 项目结构

```
hw2/
├── emotion_clustering.py      # 主分析脚本（特征提取 / 聚类 / 可视化）
├── calculate_metrics.py        # BERT 基线模型评估脚本（Google 官方）
├── analyze_data.py            # 数据集统计分析（Google 官方）
├── extract_words.py           # 情感词汇提取（Log Odds Ratio）
├── replace_emotions.py        # 情绪标签替换工具
├── report.html               # 中文 HTML 分析报告（可视化展示）
├── requirements.txt          # Python 依赖
├── data/
│   ├── train.tsv / dev.tsv / test.tsv    # 划分数据集
│   ├── emotions.txt           # 27 类情绪名称
│   ├── ekman_mapping.json     # Ekman 6 类映射
│   ├── sentiment_mapping.json # 情感极性映射
│   ├── ekman_labels.csv      # Ekman 标签
│   └── full_dataset/          # 原始 CSV（未上传，需自行下载）
├── tables/
│   ├── classification_report.xlsx   # 27 类分类指标报告
│   ├── k_scores_tfidf.xlsx         # TF-IDF K 选择指标
│   ├── k_scores_w2v.xlsx           # Word2Vec K 选择指标
│   ├── clustering_summary.xlsx     # 聚类质量汇总
│   └── emotion_words.csv           # 各情绪显著关联词
└── plots/                     # 可视化图表（运行脚本后生成）
    └── colors.tsv             # t-SNE 颜色映射数据
```

---

## 情绪分类体系

### 27 类细粒度情绪

| 类别 | 情绪词 |
|------|--------|
| Joy（喜悦） | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring |
| Anger（愤怒） | anger, annoyance, disapproval |
| Sadness（悲伤） | sadness, disappointment, embarrassment, grief, remorse |
| Fear（恐惧） | fear, nervousness |
| Disgust（厌恶） | disgust |
| Surprise（惊讶） | surprise, realization, confusion, curiosity |

### Ekman 6 类映射

本项目探索 27 类情绪能否聚合成 Ekman 的 6 类基本情绪：
`Joy` · `Anger` · `Sadness` · `Fear` · `Disgust` · `Surprise`

### 3 类情感极性

| 极性 | 说明 |
|------|------|
| Positive（积极） | joy, amusement, excitement, love, desire, optimism, caring, pride, admiration, gratitude, relief, approval |
| Negative（消极） | fear, nervousness, remorse, embarrassment, disappointment, sadness, grief, disgust, anger, annoyance, disapproval |
| Ambiguous（模糊） | realization, surprise, curiosity, confusion |

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载原始数据（可选）

```bash
mkdir -p data/full_dataset
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

### 运行完整分析

```bash
python emotion_clustering.py
```

输出图表自动保存到 `plots/`，表格保存到 `tables/`。

---

## 技术流程

```
数据加载 ── 文本预处理 ── TF-IDF / Word2Vec ── SVD 降维
    │                                          │
    │              K-Means 聚类 (K=6, K=27)    │
    │              层次聚类 (Ward 方法)        │
    │                                          │
    └────────── t-SNE 降维 ──────────────── 可视化散点图
                   │
              Logistic Regression 分类
```

### 特征提取

| 方法 | 维度 | 参数 |
|------|------|------|
| TF-IDF + SVD | 300 维 | max_features=5000, ngram=(1,2), min_df=3 |
| Word2Vec | 100 维 | window=5, min_count=3, epochs=20 |

### 聚类评估指标

- **Silhouette Score（轮廓系数）**：衡量聚类内聚度与聚类间分离度
- **Calinski-Harabasz Index（CH 指数）**：聚类间/内方差比
- **ARI（调整兰德指数）**：聚类结果与 Ekman 标签的一致性

---

## 主要发现

1. **TF-IDF 优于 Word2Vec**：词频统计特征比分布式语义表示在细粒度情绪分类中更具区分性，强指示词（如 "lol"、"hate"、"thanks"）效果显著

2. **K=6 部分对齐 Ekman**：K-Means K=6 时与 Ekman 6 类情绪有一定对齐（ARI ≈ 0.10–0.30），但 Joy 类包含 12 种细粒度情绪，合并后信息损失明显

3. **27 类不可简化为 6 类**：amusement 与 joy 在语义空间中有明显距离，说明细粒度分类具有独立价值

4. **t-SNE 揭示情绪连续性**：27 类情绪在降维空间中形成连续语义空间，相邻情绪（joy-admiration、anger-annoyance）明显重叠，支持情绪维度性假设

---

## 输出文件

### 图表（plots/）

| 文件 | 说明 |
|------|------|
| `tsne_tfidf_kmeans_k6_ekman.pdf` | TF-IDF + K=6 t-SNE（Ekman 着色） |
| `tsne_tfidf_by_ekman.pdf` | TF-IDF t-SNE（Ekman 6 类着色） |
| `tsne_tfidf_by_emotion27.pdf` | TF-IDF t-SNE（27 类情绪着色） |
| `tsne_w2v_by_ekman.pdf` | Word2Vec t-SNE（Ekman 6 类着色） |
| `k_selection_tfidf.pdf` | TF-IDF 轮廓系数 & CH 指数曲线 |
| `confusion_matrix_27.pdf` | 27 类分类混淆矩阵 |
| `svd_variance.pdf` | SVD 累计解释方差图 |
| `emotion_distribution.pdf` | 27 类情绪分布柱状图 |

### 表格（tables/）

| 文件 | 说明 |
|------|------|
| `classification_report.xlsx` | Precision / Recall / F1 per class |
| `k_scores_tfidf.xlsx` | TF-IDF K=3~15 聚类指标 |
| `clustering_summary.xlsx` | 所有实验组合汇总 |

---

## 引用

```bibtex
@inproceedings{demszky2020goemotions,
  author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
  year = {2020}
}
```

---

## 免责声明

- 数据集来源于 Reddit，可能包含固有偏差，不代表全球多样性
- 标注者均为来自印度的英语母语者，可能存在文化偏差
- 任何人使用此数据集时都应注意上述局限性
