# ocr.py
import cv2
import torch
import re
import easyocr
from config import OCR_MODEL_DIR

class SyncOCR:
    def __init__(self, regions=None):
        self.regions = regions if regions else []
        print("正在初始化 EasyOCR（本地模型）...")
        self.reader = easyocr.Reader(
            ['ch_sim', 'en'],
            model_storage_directory=OCR_MODEL_DIR,
            user_network_directory=OCR_MODEL_DIR,
            download_enabled=False,
            gpu=torch.cuda.is_available()
        )
        print("✅ EasyOCR 初始化完成")
        self.cache = {"subtitle": "", "danmaku": ""}
        self.last_processed_frame_id = -1

    def _extract_text_from_roi(self, frame, roi):
        x, y, w, h = roi
        sub_img = frame[y:y+h, x:x+w]
        if sub_img.size == 0:
            return ""
        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        if w < 300:
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result = self.reader.readtext(rgb_img, detail=1)
        texts = []
        for item in result:
            if len(item) >= 3:
                text = item[1].strip()
                conf = item[2]
                if conf > 0.15 and text:
                    texts.append(text)
        if not texts:
            return ""
        full_text = " ".join(texts)
        full_text = re.sub(r'[?？!！。，,、]', '', full_text)
        return full_text

    def update_rois(self, frame_w, frame_h):
        rois = []
        for name, (xr, yr, wr, hr), color in self.regions:
            x = int(xr * frame_w)
            y = int(yr * frame_h)
            w = int(wr * frame_w)
            h = int(hr * frame_h)
            rois.append((name, (x, y, w, h), color))
        return rois

    def recognize_frame(self, frame, frame_id, force=False):
        if not force and frame_id == self.last_processed_frame_id:
            return self.cache["subtitle"], self.cache["danmaku"]
        h, w = frame.shape[:2]
        rois = self.update_rois(w, h)
        new_texts = {}
        for name, roi, _ in rois:
            text = self._extract_text_from_roi(frame, roi)
            new_texts[name] = text if text else ""
        self.cache["subtitle"] = new_texts.get("subtitle", "")
        self.cache["danmaku"] = new_texts.get("danmaku", "")
        self.last_processed_frame_id = frame_id
        return self.cache["subtitle"], self.cache["danmaku"]