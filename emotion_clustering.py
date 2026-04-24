# coding=utf-8
"""
GoEmotions 27类情绪分类 · 监督学习 + 聚类分析 + 可视化

任务:
1. TF-IDF / Word2Vec 特征提取
2. PCA (SVD) 降维
3. K-Means / 层次聚类  (探索 27类情绪能否聚成 Ekman 的 6 基本情绪)
4. t-SNE / UMAP 可视化
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    from gensim.models import Word2Vec as GW
    GENSIM_AVAILABLE = True
except ImportError:
    GW = None
    GENSIM_AVAILABLE = False
    print("[INFO] gensim not available — using CountVectorizer fallback for word embeddings")
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk

warnings.filterwarnings("ignore")

# ─────────────────────────── 配置 ───────────────────────────

DATA_DIR   = Path("data")
PLOTS_DIR  = Path("plots")
TABLES_DIR = Path("tables")

PLOTS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

EMOTION_FILE = DATA_DIR / "emotions.txt"
EKMAN_FILE   = DATA_DIR / "ekman_mapping.json"
SENTIMENT_FILE = DATA_DIR / "sentiment_mapping.json"

RANDOM_STATE = 42

# 27 维 Ekman 情绪映射
EKMAN_MAP = {
    "anger":     ["anger", "annoyance", "disapproval"],
    "disgust":   ["disgust"],
    "fear":      ["fear", "nervousness"],
    "joy":       ["joy", "amusement", "approval", "excitement", "gratitude",
                  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness":   ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise":  ["surprise", "realization", "confusion", "curiosity"],
}

EKMAN_COLORS = {
    "anger":    "#E74C3C",
    "disgust":  "#8E44AD",
    "fear":     "#2C3E50",
    "joy":      "#F39C12",
    "sadness":  "#3498DB",
    "surprise": "#1ABC9C",
}

SENTIMENT_COLORS = {
    "positive": "#27AE60",
    "negative": "#E74C3C",
    "ambiguous": "#F39C12",
}

# ─────────────────────────── 工具函数 ───────────────────────────

def load_data():
    """加载 train / dev / test 数据"""
    def load_split(path):
        cols = ["text", "labels", "id"]
        df = pd.read_csv(path, sep="\t", header=None, names=cols, encoding="utf-8")
        return df

    train = load_split(DATA_DIR / "train.tsv")
    dev   = load_split(DATA_DIR / "dev.tsv")
    test  = load_split(DATA_DIR / "test.tsv")

    with open(EMOTION_FILE, "r", encoding="utf-8") as f:
        emotions = [e.strip() for e in f.read().splitlines()]

    df = pd.concat([train, dev, test], ignore_index=True)
    print(f"总样本数: {len(df)}")
    print(f"情感类别数: {len(emotions)}")
    return df, emotions


def preprocess_text(text):
    """基础文本清洗"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_label_mapping(df, emotions):
    """将逗号分隔的标签转为单一主标签（出现最多次）"""
    label_counts = {e: i for i, e in enumerate(emotions)}
    primary_labels = []
    ekman_labels   = []

    for _, row in df.iterrows():
        labels = str(row["labels"]).split(",")
        label_list = [int(l.strip()) for l in labels if l.strip().isdigit()]
        if not label_list:
            primary_labels.append(-1)
            ekman_labels.append("unknown")
            continue

        counts = {}
        for l in label_list:
            counts[l] = counts.get(l, 0) + 1
        primary = max(counts, key=counts.get)
        primary_labels.append(primary)

        emotion_name = emotions[primary] if primary < len(emotions) else "unknown"
        ek = "unknown"
        for ek_name, ek_emotions in EKMAN_MAP.items():
            if emotion_name in ek_emotions:
                ek = ek_name
                break
        ekman_labels.append(ek)

    df = df.copy()
    df["primary_label"] = primary_labels
    df["primary_emotion"] = df["primary_label"].apply(
        lambda x: emotions[x] if 0 <= x < len(emotions) else "unknown"
    )
    df["ekman_label"] = ekman_labels

    # 过滤 neutral (27)
    df_clean = df[df["primary_label"] != 27].copy()
    print(f"过滤 neutral 后样本数: {len(df_clean)}")
    return df_clean


def get_ekman_colors_for_emotions(emotion_names, emotions):
    colors = []
    for e in emotion_names:
        for ek_name, ek_list in EKMAN_MAP.items():
            if e in ek_list:
                colors.append(EKMAN_COLORS[ek_name])
                break
        else:
            colors.append("#95A5A6")
    return colors


def get_sentiment_colors_for_emotions(emotion_names):
    with open(SENTIMENT_FILE, "r", encoding="utf-8") as f:
        sent_map = json.load(f)
    sent_rev = {}
    for k, vs in sent_map.items():
        for v in vs:
            sent_rev[v] = k
    colors = [SENTIMENT_COLORS.get(sent_rev.get(e, "ambiguous"), "#95A5A6") for e in emotion_names]
    return colors


# ─────────────────────────── 特征提取 ───────────────────────────

def extract_tfidf(corpus, n_components=300, ngram_range=(1, 2), max_features=5000):
    """TF-IDF 特征提取 + SVD 降维"""
    print("\n[TF-IDF] 构建词袋向量...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(corpus)
    print(f"  TF-IDF 维度: {X_tfidf.shape}")

    print(f"[SVD] 降维到 {n_components} 维...")
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_tfidf)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD 累计解释方差: {explained:.2%}")
    return X_svd, vectorizer, svd


def extract_word2vec(corpus, embedding_dim=100, window=5, min_count=3):
    """Word2Vec 词嵌入（对句子取平均得到文档向量）"""
    print("\n[Word2Vec] 训练词嵌入...")
    tokenized = [text.split() for text in corpus]

    if GENSIM_AVAILABLE:
        model = GW(
            sentences=tokenized,
            vector_size=embedding_dim,
            window=window,
            min_count=min_count,
            workers=4,
            seed=RANDOM_STATE,
            epochs=20,
        )
        print(f"  词汇表大小: {len(model.wv)}")
        doc_vectors = []
        for tokens in tokenized:
            vecs = [model.wv[w] for w in tokens if w in model.wv]
            if vecs:
                doc_vectors.append(np.mean(vecs, axis=0))
            else:
                doc_vectors.append(np.zeros(embedding_dim))
        X_w2v = np.array(doc_vectors)
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import TruncatedSVD as SVD2
        cv = CountVectorizer(max_features=5000, min_df=min_count)
        X_cnt = cv.fit_transform(corpus)
        svd2 = SVD2(n_components=embedding_dim, random_state=RANDOM_STATE)
        X_w2v = svd2.fit_transform(X_cnt)
        print(f"  (fallback SVD-based) 维度: {X_w2v.shape}")

    print(f"  Word2Vec 文档向量维度: {X_w2v.shape}")
    return X_w2v, model if GENSIM_AVAILABLE else None


# ─────────────────────────── 聚类分析 ───────────────────────────

def find_optimal_k(X, k_range, metric="cosine"):
    """轮廓系数找最优 K"""
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric=metric)
        ch = calinski_harabasz_score(X, labels)
        scores.append({"k": k, "silhouette": score, "calinski_harabasz": ch})
        print(f"  K={k:2d}  轮廓系数={score:.4f}  CH指数={ch:.1f}")
    return pd.DataFrame(scores)


def kmeans_clustering(X, n_clusters, labels_true=None):
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    y_pred = km.fit_predict(X)
    sil = silhouette_score(X, y_pred, metric="cosine")
    ch  = calinski_harabasz_score(X, y_pred)
    print(f"\n[K-Means K={n_clusters}] 轮廓系数={sil:.4f}  CH指数={ch:.1f}")

    if labels_true is not None:
        ari = adjusted_rand_score(labels_true, y_pred)
        print(f"  ARI (vs Ekman) = {ari:.4f}")

    return y_pred, km, sil, ch


def hierarchical_clustering(X, n_clusters, linkage_method="ward"):
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric="euclidean",
    )
    y_pred = agg.fit_predict(X)
    sil = silhouette_score(X, y_pred, metric="cosine")
    ch  = calinski_harabasz_score(X, y_pred)
    print(f"\n[层次聚类 K={n_clusters}] 轮廓系数={sil:.4f}  CH指数={ch:.1f}")
    return y_pred, sil, ch


def cluster_emotion_centroids(X, y_pred, emotion_names, emotions):
    """计算每个聚类中心的情感分布"""
    n_clusters = len(np.unique(y_pred))
    results = []
    for c in range(n_clusters):
        mask = y_pred == c
        if mask.sum() == 0:
            continue
        cluster_emotions = [emotion_names[i] for i in np.where(mask)[0]]
        from collections import Counter
        cnt = Counter(cluster_emotions)
        results.append({"cluster": c, "size": mask.sum(), "top_emotions": cnt.most_common(3)})
        print(f"  Cluster {c} ({mask.sum()} samples): {cnt.most_common(5)}")
    return results


# ─────────────────────────── 监督学习 ───────────────────────────

def supervised_tfidf(X_train, y_train, X_test, y_test, label_names):
    print("\n[监督学习] Logistic Regression on TF-IDF...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"  准确率: {acc:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report).T.round(4)
    report_df.to_excel(TABLES_DIR / "classification_report.xlsx")
    print("  分类报告已保存到 tables/classification_report.xlsx")
    return clf, acc, report_df


# ─────────────────────────── 可视化 ───────────────────────────

def plot_silhouette_scores(df_scores, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df_scores["k"], df_scores["silhouette"], "bo-", lw=2)
    axes[0].set_xlabel("K (Number of Clusters)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title(f"{title} — Silhouette Score")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_scores["k"], df_scores["calinski_harabasz"], "rs-", lw=2)
    axes[1].set_xlabel("K (Number of Clusters)")
    axes[1].set_ylabel("Calinski-Harabasz Index")
    axes[1].set_title(f"{title} — Calinski-Harabasz Index")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


def plot_tsne(X_2d, labels, title, fname, palette=None, figsize=(14, 12)):
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="tab20", alpha=0.6, s=8
    )
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.colorbar(scatter, label="Cluster / Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


def plot_tsne_by_ekman(X_2d, ekman_labels, title, fname, figsize=(16, 12)):
    unique_ek = sorted(set(ekman_labels))
    color_map = {e: EKMAN_COLORS.get(e, "#95A5A6") for e in unique_ek}
    colors = [color_map.get(e, "#95A5A6") for e in ekman_labels]

    plt.figure(figsize=figsize)
    for ek in unique_ek:
        mask = [e == ek for e in ekman_labels]
        label = ek.capitalize() if ek != "unknown" else "Unknown"
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color_map.get(ek, "#95A5A6"),
            label=label,
            alpha=0.65, s=10,
        )
    plt.legend(title="Ekman Emotion", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


def plot_tsne_by_emotion(X_2d, emotion_names, emotions, title, fname, n_show=27, figsize=(18, 16)):
    unique = list(set(emotion_names))
    n_colors = len(unique)
    cmap = plt.cm.get_cmap("tab20", n_colors)
    color_map = {e: cmap(i % 20) for i, e in enumerate(unique)}

    plt.figure(figsize=figsize)
    for em in unique:
        mask = [e == em for e in emotion_names]
        if sum(mask) == 0:
            continue
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color_map[em]], label=em.capitalize(),
            alpha=0.6, s=10,
        )
    plt.legend(
        title="27 Emotions", bbox_to_anchor=(1.02, 1),
        loc="upper left", fontsize=7, ncol=1,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


def plot_confusion_matrix(y_true, y_pred, label_names, title, fname, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1e-9)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, annot=False, fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=[e.capitalize() for e in label_names],
        yticklabels=[e.capitalize() for e in label_names],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


def plot_pca_variance(svd, fname):
    cumvar = np.cumsum(svd.explained_variance_ratio_)
    n = len(cumvar)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, n+1), svd.explained_variance_ratio_, alpha=0.6, color="steelblue")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("SVD — Single Component Variance")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, n+1), cumvar, "rs-", lw=2)
    axes[1].axhline(y=0.8, color="gray", linestyle="--", label="80%")
    axes[1].axhline(y=0.9, color="gray", linestyle=":", label="90%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("SVD — Cumulative Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


def plot_emotion_distribution(df, fname):
    counts = df["primary_emotion"].value_counts()
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(counts.index, counts.values, color="steelblue", alpha=0.8)
    ek_colors = get_ekman_colors_for_emotions(counts.index, None)
    for bar, color in zip(bars, ek_colors):
        bar.set_color(color)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_title("27-Class Emotion Distribution")
    plt.xticks(rotation=45, ha="right")
    for i, (idx, val) in enumerate(counts.items()):
        ax.text(i, val + 50, str(val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


def plot_ekman_cluster_matrix(y_pred, ekman_labels_true, n_clusters, fname):
    """聚类 vs Ekman 真实标签的交叉热力图"""
    if len(y_pred) != len(ekman_labels_true):
        print(f"  跳过聚类热力图（长度不匹配 {len(y_pred)} vs {len(ekman_labels_true)}）")
        return
    df_cross = pd.crosstab(
        pd.Series(ekman_labels_true, name="True Ekman"),
        pd.Series(y_pred, name="Cluster"),
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        df_cross,
        annot=True, fmt="d",
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_title("Ekman Emotion vs Cluster (K-Means)")
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("Ekman Emotion")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {PLOTS_DIR / fname}")


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    plt.rcParams["font.size"] = 10
    plt.rcParams["figure.dpi"] = 150

    print("=" * 70)
    print("GoEmotions 27类情绪分类 · 聚类分析 + 可视化")
    print("=" * 70)

    # ── 1. 数据加载 ──────────────────────────────────────────────
    print("\n[1] 加载数据...")
    df_raw, emotions = load_data()

    # ── 2. 文本预处理 ────────────────────────────────────────────
    print("\n[2] 预处理文本...")
    df = build_label_mapping(df_raw, emotions)
    df["clean_text"] = df["text"].apply(preprocess_text)

    # 过滤空文本
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"  有效样本: {len(df)}")

    # ── 3. 特征提取 ─────────────────────────────────────────────
    print("\n[3] 特征提取...")

    X_tfidf, vec_tfidf, svd_tfidf = extract_tfidf(
        df["clean_text"].tolist(),
        n_components=300,
        max_features=5000,
    )
    X_w2v, model_w2v = extract_word2vec(
        df["clean_text"].tolist(),
        embedding_dim=100,
    )

    # ── 4. PCA/SVD 方差分析 ──────────────────────────────────────
    print("\n[4] SVD 方差分析...")
    plot_pca_variance(svd_tfidf, "svd_variance.pdf")

    # ── 5. t-SNE 降维（用于可视化）──────────────────────────────
    print("\n[5] t-SNE 降维（可视化用，采样3000点）...")
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_tfidf), size=min(3000, len(X_tfidf)), replace=False
    )
    X_tfidf_sample = X_tfidf[sample_idx]
    X_w2v_sample   = X_w2v[sample_idx]
    df_sample      = df.iloc[sample_idx].reset_index(drop=True)

    print("  t-SNE on TF-IDF...")
    tsne_tfidf = TSNE(
        n_components=2, perplexity=30, random_state=RANDOM_STATE,
        n_iter=1000, learning_rate="auto", init="pca",
    )
    X_tsne_tfidf = tsne_tfidf.fit_transform(X_tfidf_sample)

    print("  t-SNE on Word2Vec...")
    tsne_w2v = TSNE(
        n_components=2, perplexity=30, random_state=RANDOM_STATE,
        n_iter=1000, learning_rate="auto", init="pca",
    )
    X_tsne_w2v = tsne_w2v.fit_transform(X_w2v_sample)

    # ── 6. 监督学习：TF-IDF + Logistic Regression ─────────────
    print("\n[6] 监督学习实验...")
    emotion_names = df["primary_emotion"].values
    ekman_names   = df["ekman_label"].values

    le = LabelEncoder()
    y_all = le.fit_transform(emotion_names)
    label_names = le.classes_.tolist()

    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y_all)), y_all,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_all,
    )
    X_train_tfidf = X_tfidf[X_train_idx]
    X_test_tfidf  = X_tfidf[X_test_idx]

    clf, acc, report_df = supervised_tfidf(
        X_train_tfidf, y_train, X_test_tfidf, y_test, label_names
    )
    y_pred_test = clf.predict(X_test_tfidf)

    # 混淆矩阵
    plot_confusion_matrix(
        y_test, y_pred_test, label_names,
        "Confusion Matrix — TF-IDF + Logistic Regression (27 emotions)",
        "confusion_matrix_27.pdf", normalize=False,
    )

    # ── 7. 聚类分析 ─────────────────────────────────────────────
    print("\n[7] 聚类分析...")

    k_range = range(3, 16)

    # 7a TF-IDF 聚类
    print("\n  [TF-IDF] 寻找最优 K...")
    df_scores_tfidf = find_optimal_k(X_tfidf_sample, k_range)
    df_scores_tfidf.to_excel(TABLES_DIR / "k_scores_tfidf.xlsx", index=False)
    plot_silhouette_scores(df_scores_tfidf, "TF-IDF + SVD", "k_selection_tfidf.pdf")

    # 7b Word2Vec 聚类
    print("\n  [Word2Vec] 寻找最优 K...")
    df_scores_w2v = find_optimal_k(X_w2v_sample, k_range)
    df_scores_w2v.to_excel(TABLES_DIR / "k_scores_w2v.xlsx", index=False)
    plot_silhouette_scores(df_scores_w2v, "Word2Vec", "k_selection_w2v.pdf")

    # 7c K=6 vs K=27 对比（Ekman 6 vs 原始 27）
    for feat_name, X_full, X_samp in [
        ("TF-IDF", X_tfidf, X_tfidf_sample),
        ("Word2Vec", X_w2v, X_w2v_sample),
    ]:
        ekman_le = LabelEncoder()
        y_ekman  = ekman_le.fit_transform(df["ekman_label"].values)
        ekman_labels_sample = y_ekman[sample_idx]

        # Ekman 6 类
        for n_clust, tag in [(6, "Ekman6"), (27, "Full27")]:
            print(f"\n  [{feat_name}] K-Means K={n_clust} ({tag})")
            y_km, km_model, sil, ch = kmeans_clustering(X_samp, n_clust, ekman_labels_sample)

            # t-SNE 可视化
            y_col = y_km
            if n_clust == 6:
                title_ek = f"[{feat_name}] K-Means K=6 (Ekman) — t-SNE"
                plot_tsne(
                    X_tsne_tfidf if feat_name == "TF-IDF" else X_tsne_w2v,
                    y_col, title_ek,
                    f"tsne_{feat_name.lower()}_kmeans_k{n_clust}_ekman.pdf",
                    palette=EKMAN_COLORS,
                )
                plot_ekman_cluster_matrix(y_km, ekman_le.inverse_transform(ekman_labels_sample), n_clust,
                    f"cluster_ekman_heatmap_{feat_name.lower()}_k{n_clust}.pdf")

            # 层次聚类
            y_agg, sil_agg, ch_agg = hierarchical_clustering(X_samp, n_clust)
            print(f"    层次聚类 轮廓={sil_agg:.4f}  CH={ch_agg:.1f}")

    # ── 8. t-SNE 综合可视化 ────────────────────────────────────
    print("\n[8] 综合可视化...")

    ekman_labels_sample_str = [
        ekman_le.inverse_transform([y])[0] if hasattr(ekman_le, "inverse_transform") else ""
        for y in ekman_labels_sample
    ]
    # 重新计算 ekman 编码
    ekman_map_inv = {}
    for ek, elist in EKMAN_MAP.items():
        for e in elist:
            ekman_map_inv[e] = ek
    ekman_sample = [ekman_map_inv.get(e, "unknown") for e in df_sample["primary_emotion"].values]

    # 8a 按 Ekman 6 类着色
    plot_tsne_by_ekman(
        X_tsne_tfidf, ekman_sample,
        "[TF-IDF] t-SNE 情绪分布 (按 Ekman 6 类着色)",
        "tsne_tfidf_by_ekman.pdf",
    )
    plot_tsne_by_ekman(
        X_tsne_w2v, ekman_sample,
        "[Word2Vec] t-SNE 情绪分布 (按 Ekman 6 类着色)",
        "tsne_w2v_by_ekman.pdf",
    )

    # 8b 按 27 类着色
    plot_tsne_by_emotion(
        X_tsne_tfidf, df_sample["primary_emotion"].values, emotions,
        "[TF-IDF] t-SNE 情绪分布 (按 27 类着色)",
        "tsne_tfidf_by_emotion27.pdf",
    )
    plot_tsne_by_emotion(
        X_tsne_w2v, df_sample["primary_emotion"].values, emotions,
        "[Word2Vec] t-SNE 情绪分布 (按 27 类着色)",
        "tsne_w2v_by_emotion27.pdf",
    )

    # ── 9. 情绪分布 & 统计报告 ──────────────────────────────────
    print("\n[9] 统计报告...")
    plot_emotion_distribution(df, "emotion_distribution.pdf")

    # 聚类中心分析
    print("\n  聚类中心情感分布 (K=6, TF-IDF):")
    y_k6, _, _, _ = kmeans_clustering(X_tfidf_sample, 6)
    cluster_emotion_centroids(X_tfidf_sample, y_k6, df_sample["primary_emotion"].values, emotions)

    # 聚类质量汇总表
    summary = []
    for feat_name, X_samp in [("TF-IDF+SVD", X_tfidf_sample), ("Word2Vec", X_w2v_sample)]:
        for n_clust in [6, 27]:
            km = KMeans(n_clusters=n_clust, random_state=RANDOM_STATE, n_init=10)
            y_p = km.fit_predict(X_samp)
            sil = silhouette_score(X_samp, y_p, metric="cosine")
            ch  = calinski_harabasz_score(X_samp, y_p)
            if n_clust == 6:
                ari = adjusted_rand_score(ekman_labels_sample, y_p)
            else:
                ari = np.nan
            summary.append({
                "Feature": feat_name,
                "K": n_clust,
                "Silhouette": round(sil, 4),
                "Calinski-Harabasz": round(ch, 1),
                "ARI_vs_Ekman": round(ari, 4) if not np.isnan(ari) else "-",
            })
    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(TABLES_DIR / "clustering_summary.xlsx", index=False)
    print("\n聚类汇总:")
    print(df_summary.to_string(index=False))
    print(f"\n  已保存: {TABLES_DIR / 'clustering_summary.xlsx'}")

    print("\n" + "=" * 70)
    print("全部任务完成！图表保存在 plots/，表格保存在 tables/")
    print("=" * 70)


if __name__ == "__main__":
    main()
