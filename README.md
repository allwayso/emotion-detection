# 😡🤢😱😊😐😔😲 Emotion Detection — EfficientNetB0 Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 **EfficientNetB0 迁移学习**的面部情绪检测模型，使用 FER-2013 数据集训练，支持 7 类情绪识别。

Facial emotion detection based on **EfficientNetB0 transfer learning**, trained on the FER-2013 dataset with 7-class emotion classification.

---

## 📑 目录 / Table of Contents

- [功能特性 / Features](#-功能特性--features)
- [情绪类别 / Emotion Classes](#-情绪类别--emotion-classes)
- [模型架构 / Model Architecture](#-模型架构--model-architecture)
- [项目结构 / Project Structure](#-项目结构--project-structure)
- [环境要求 / Requirements](#-环境要求--requirements)
- [快速开始 / Quick Start](#-快速开始--quick-start)
- [训练流程 / Training Pipeline](#-训练流程--training-pipeline)
- [模型评估 / Evaluation](#-模型评估--evaluation)
- [命令行预测 / CLI Demo](#-命令行预测--cli-demo)
- [配置参数 / Hyperparameters](#-配置参数--hyperparameters)
- [License](#-license)

---

## ✨ 功能特性 / Features

- 🧠 **EfficientNetB0** 骨干网络 + ImageNet 预训练权重
- 🎯 7 类情绪分类：Anger, Disgust, Fear, Happy, Neutral, Sadness, Surprise
- ⚡ **Mixed Precision FP16** 训练加速（~2x on NVIDIA P100）
- 📈 **两阶段训练策略**：冻结骨干 → 微调最后 20% 层
- 🔧 **类别权重**处理数据不平衡
- 📊 完整的评估指标：混淆矩阵、分类报告、ROC 曲线
- 🖥️ 命令行单张图像预测 Demo
- 📦 极轻量：总参数量仅 ~4.5M

---

## 🎭 情绪类别 / Emotion Classes

| 索引 | Emoji | 情绪 (English) | 情绪 (中文) |
|------|-------|----------------|-------------|
| 0    | 😡    | Anger          | 愤怒        |
| 1    | 🤢    | Disgust        | 厌恶        |
| 2    | 😱    | Fear           | 恐惧        |
| 3    | 😊    | Happy          | 开心        |
| 4    | 😐    | Neutral        | 中性        |
| 5    | 😔    | Sadness        | 悲伤        |
| 6    | 😲    | Surprise       | 惊讶        |

---

## 🏗️ 模型架构 / Model Architecture

```
Input (48×48×1)
    │
    ▼
Conv2D (3, 1×1)          ← 灰度 → RGB 适配层 / Grayscale to RGB adapter
    │
    ▼
UpSampling2D (2×)        ← 48×48 → 96×96 上采样 / Upsample for EfficientNet
    │
    ▼
EfficientNetB0            ← ImageNet 预训练骨干 / Pretrained backbone
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(256, Swish)         ← 全连接层
    │
    ▼
BatchNormalization
    │
    ▼
Dropout(0.4)
    │
    ▼
Dense(7, Softmax)         ← 7 类输出 / 7-class output
```

| 组件 / Component | 配置 / Configuration |
|-------------------|----------------------|
| 骨干网络 / Backbone | EfficientNetB0 (ImageNet pretrained) |
| 输入尺寸 / Input Size | 48×48 灰度图 / grayscale |
| 分类头 / Classification Head | GAP → Dense(256, Swish) → Dropout(0.4) → Softmax(7) |
| 优化器 / Optimizer | AdamW + Warmup + Cosine Decay |
| 正则化 / Regularization | Label Smoothing + L2 + Dropout + Class Weight |
| 总参数量 / Total Params | ~4.5M |

---

## 📁 项目结构 / Project Structure

```
emotion-detection/
├── emotion-detection-new.ipynb   # Jupyter Notebook：完整训练+评估流程
├── demo.py                       # 命令行推理脚本 / CLI inference script
├── best_s2.keras                 # 第二阶段微调后的最佳模型
├── README.md                     # 项目说明（本文件）
└── LICENSE                       # 开源协议
```

---

## 📋 环境要求 / Requirements

- **Python** ≥ 3.10
- **TensorFlow** ≥ 2.x (推荐 2.15+)
- **CUDA** (可选，GPU 训练推荐)

### Python 依赖 / Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn plotly opencv-python scikit-learn
```

---

## 🚀 快速开始 / Quick Start

### 1. 克隆仓库 / Clone the repo

```bash
git clone git@github.com:allwayso/emotion-detection.git
cd emotion-detection
```

### 2. 安装依赖 / Install dependencies

```bash
pip install -r <(echo -e "tensorflow\nnumpy\npandas\nmatplotlib\nseaborn\nplotly\nopencv-python\nscikit-learn")
```

> 💡 也可直接运行 `pip install tensorflow numpy pandas matplotlib seaborn plotly opencv-python scikit-learn`。

### 3. 命令行预测 / CLI Prediction

```bash
python demo.py <图像路径>
```

示例 / Example：

```bash
python demo.py face.jpg
```

输出示例 / Sample output：

```
📦 加载模型: best_s2.keras ...
✅ 模型加载完成

🎯 预测结果: 😊 Happy  (94.5%)
---------------------------------------------
  😡 Anger      0.0023  ▌
  🤢 Disgust    0.0011  
  😱 Fear       0.0056  ▎
  😊 Happy      0.9447  ████████████████████████████████████████  ←
  😐 Neutral    0.0312  █▎
  😔 Sadness    0.0104  ▌
  😲 Surprise   0.0047  ▎
```

### 4. 完整训练（Notebook）

在 Jupyter 环境中打开 `emotion-detection-new.ipynb`，按顺序执行各 Cell。

> ⚠️ 注意：数据集路径需指向 FER-2013 的 `train/` 和 `test/` 目录。

---

## 🔄 训练流程 / Training Pipeline

训练分为两个阶段：

### 第一阶段：冻结骨干 / Stage 1 — Frozen Backbone

| 参数 | 值 |
|------|-----|
| 可训练部分 | 灰度适配层 + 分类头 |
| 学习率 | 1e-3 (Warmup + Cosine Decay) |
| Batch Size | 128 |
| 最大 Epoch | 30 (EarlyStopping patience=5) |
| 目的 | 训练分类头，保护预训练特征 |

### 第二阶段：微调 / Stage 2 — Fine-tuning

| 参数 | 值 |
|------|-----|
| 可训练部分 | 骨干最后 20% 层 + 分类头 |
| 学习率 | 5e-5 (ReduceLROnPlateau) |
| Batch Size | 128 |
| 最大 Epoch | 30 (EarlyStopping patience=3) |
| 目的 | 微调高级特征，适应表情识别 |

---

## 📊 模型评估 / Evaluation

Notebook 中包含完整的评估流程：

- **混淆矩阵 / Confusion Matrix** — 热力图展示各类别预测分布
- **分类报告 / Classification Report** — Precision, Recall, F1-Score
- **ROC 曲线 / ROC Curves** — 各类别 AUC 值 + Macro/Weighted Avg

### 数据增强 / Data Augmentation（仅训练集）

| 增强方式 | 参数 |
|----------|------|
| 随机旋转 / Rotation | ±10° |
| 随机缩放 / Zoom | ±10% |
| 水平翻转 / Horizontal Flip | ✅ |
| 亮度微调 / Brightness | [0.9, 1.1] |

---

## 🖥️ 命令行预测 / CLI Demo

`demo.py` 支持直接传入图像路径进行预测：

```bash
python demo.py <image_path>
```

特性：
- 自动加载 `best_s2.keras` 模型
- 支持任意尺寸输入，自动缩放为 48×48 灰度图
- 输出 Top-1 预测结果 + 所有类别概率柱状图
- 彩色 emoji 标记，直观易读

---

## ⚙️ 配置参数 / Hyperparameters

| 参数 / Parameter | 值 / Value | 说明 / Description |
|------------------|------------|---------------------|
| `IMG_SIZE` | 48 | 输入图像尺寸 |
| `BATCH_SIZE` | 128 | 批次大小 |
| `EPOCHS_S1` | 30 | 第一阶段最大 epoch |
| `EPOCHS_S2` | 30 | 第二阶段最大 epoch |
| `LR_S1_MAX` | 1e-3 | 第一阶段学习率 |
| `LR_S2_MAX` | 5e-5 | 第二阶段学习率 |
| `DROPOUT` | 0.4 | Dropout 比例 |
| `L2_REG` | 1e-4 | L2 正则化系数 |
| `LABEL_SMTH` | 0.1 | Label Smoothing |
| `FT_PERCENT` | 0.2 | 微调时解冻比例 |
| `WARMUP` | 3 | Warmup epoch 数 |
| `SEED` | 42 | 随机种子 |

---

## 📄 License

MIT License © 2025

---

## 🙏 致谢 / Acknowledgments

- **数据集 / Dataset**: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) — Facial Expression Recognition 2013
- **骨干网络 / Backbone**: [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, ICML 2019
- **训练环境 / Platform**: Kaggle Notebook (NVIDIA P100 GPU)

---

<p align="center">
  <sub>Made with ❤️ for emotion AI research</sub>
</p>