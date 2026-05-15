"""
情绪检测 Demo — 命令行单张图像预测
用法: python demo.py <图像路径>
示例: python demo.py face.jpg
"""

import sys
import numpy as np
import cv2
import tensorflow as tf

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "best_s2.keras"
CLS = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]
EMJ = ["😡", "🤢", "😱", "😊", "😐", "😔", "😲"]


def load_model():
    print(f"📦 加载模型: {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ 模型加载完成")
    return model


def predict(image_path, model):
    # 1. 读取为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 2. 缩放到 48×48
    img = cv2.resize(img, (48, 48))

    # 3. 转 float32（保持 [0, 255]，训练时无 rescale）
    img = img.astype(np.float32)

    # 4. 加 batch + channel 维度 → (1, 48, 48, 1)
    img = np.expand_dims(img, axis=(0, -1))

    # 5. 推理
    probs = model.predict(img, verbose=0)[0]
    pred_idx = np.argmax(probs)

    # 6. 打印结果
    print(f"\n🎯 预测结果: {EMJ[pred_idx]} {CLS[pred_idx]}  ({probs[pred_idx]:.1%})")
    print("-" * 45)
    for i, (emo, label) in enumerate(zip(EMJ, CLS)):
        bar = "█" * int(probs[i] * 40)
        mark = "  ←" if i == pred_idx else ""
        print(f"  {emo} {label:8s}  {probs[i]:.4f}  {bar}{mark}")


def main():
    if len(sys.argv) < 2:
        print("用法: python demo.py <图像路径>")
        print("示例: python demo.py face.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()
    predict(image_path, model)


if __name__ == "__main__":
    main()