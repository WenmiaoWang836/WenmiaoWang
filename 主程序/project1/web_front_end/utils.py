# utils.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import FONT_PATH

def draw_chinese_text(img, text, pos, font_size=22, color=(255, 255, 255)):
    try:
        font = ImageFont.truetype(FONT_PATH, font_size, encoding="utf-8")
    except:
        # 回退到默认字体
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size/20, color, 2)
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)