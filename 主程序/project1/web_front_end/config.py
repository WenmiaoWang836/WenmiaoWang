# config.py
import os

# 模型文件路径（请修改为实际路径）
TEXT_MODEL_PATH = r"E:\python\project1\best_danmu_model_v1.pth"
VISUAL_MODEL_PATH = r"E:\python\project1\best_emotion_model.pth"
OCR_MODEL_DIR = r"E:\python\project1"          # EasyOCR 模型存放目录
YU_NET_MODEL_PATH = r"E:\python\project1\face_detection_yunet_2023mar.onnx"

# ROI 比例（相对于帧尺寸）
FACE_ROI_RATIO = (0, 0, 1, 1)          # (x_ratio, y_ratio, w_ratio, h_ratio)
TEXT_REGIONS = [
    ("subtitle", (0, 0.7, 0.8, 0.3), (255, 0, 0)),   # 底部字幕
    ("danmaku",  (0.7, 0,   0.3, 0.7), (0, 255, 255)) # 右侧弹幕
]

# 中文绘制字体（Windows 默认，Linux 请修改）
FONT_PATH = r"C:/Windows/Fonts/simhei.ttf"