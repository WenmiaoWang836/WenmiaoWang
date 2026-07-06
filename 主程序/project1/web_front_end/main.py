# main.py - 生成器版本（实时流）
import os
import cv2
import time
import gradio as gr
from config import TEXT_MODEL_PATH, VISUAL_MODEL_PATH, FACE_ROI_RATIO, TEXT_REGIONS
from fusion import MultiModalEmotionFusion
from ocr import SyncOCR
from utils import draw_chinese_text

fusion = MultiModalEmotionFusion(TEXT_MODEL_PATH, VISUAL_MODEL_PATH)

def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------- 新增：缩放帧 ----------
    SCALE = 0.7   # 可调，建议 0.5 ~ 0.8
    if SCALE != 1.0:
        width = int(width * SCALE)
        height = int(height * SCALE)

    # ROI 仍按比例计算（config.py 中比例不变）
    roi_x = int(width * FACE_ROI_RATIO[0])
    roi_y = int(height * FACE_ROI_RATIO[1])
    roi_w = int(width * FACE_ROI_RATIO[2])
    roi_h = int(height * FACE_ROI_RATIO[3])
    face_roi = (roi_x, roi_y, roi_w, roi_h)

    ocr = SyncOCR(regions=TEXT_REGIONS)
    frame_counter = 0
    ocr_interval = 15     # 增大 OCR 间隔
    face_interval = 5    # 增大人脸检测间隔
    frame_time = 1.0 / fps if fps > 0 else 0.033

    # ---------- 用于缓存上一帧结果 ----------
    last_display = "等待处理..."
    last_subtitle = ""
    last_danmaku = ""

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1

        # 缩放帧
        if SCALE != 1.0:
            frame = cv2.resize(frame, (width, height))

        # ---- 仅在关键帧做 OCR 和预测 ----
        do_ocr = (frame_counter % ocr_interval == 0) or (frame_counter == 1)
        do_face = (frame_counter % face_interval == 0) or (frame_counter == 1)

        if do_ocr:
            subtitle, danmaku = ocr.recognize_frame(frame, frame_counter, force=True)
            current_text = (subtitle + " " + danmaku).strip()
            # 保存最新文本
            last_subtitle = subtitle
            last_danmaku = danmaku
        else:
            # 复用上一次的文本
            subtitle = last_subtitle
            danmaku = last_danmaku
            current_text = (subtitle + " " + danmaku).strip()

        # 预测（仅在关键帧做）
        if do_face or do_ocr:
            if current_text:
                result = fusion.predict_multimodal(current_text, frame, face_roi, force_face=do_face, frame_id=frame_counter)
                if result['emotion'] == 'No face':
                    display = f"无人脸 | 文本: {current_text[:20]}"
                else:
                    display = f"{result['emotion']} ({result['confidence']*100:.1f}%)"
            else:
                vis_emotion, vis_conf, _ = fusion.predict_visual(frame, face_roi, force_face=do_face, frame_id=frame_counter)
                if vis_emotion:
                    display = f"仅视觉: {vis_emotion} ({vis_conf*100:.1f}%)"
                else:
                    display = "未检测到人脸"
            # 更新缓存
            last_display = display
        else:
            # 非关键帧直接使用上一次预测结果
            display = last_display

        # ---- 绘制（每帧都绘制，但使用缓存的信息） ----
        # 绘制 ROI 框
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 0, 255), 2)
        cv2.putText(frame, "Face ROI", (roi_x+5, roi_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        text_rois = ocr.update_rois(width, height)
        for name, (x, y, w, h), color in text_rois:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        fusion.draw_faces(frame, face_roi, frame_id=frame_counter)

        # 中文显示（使用缓存文本）
        frame = draw_chinese_text(frame, f"Emotion: {display}", (10, 30), 24, (0, 255, 0))
        y_offset = 70
        if subtitle:
            frame = draw_chinese_text(frame, f"字幕: {subtitle[:40]}", (10, y_offset), 22, (255, 255, 0))
            y_offset += 30
        if danmaku:
            frame = draw_chinese_text(frame, f"弹幕: {danmaku[:40]}", (10, y_offset), 22, (255, 255, 0))
            y_offset += 30
        frame = draw_chinese_text(frame, display, (10, y_offset+5), 18, (0, 255, 255))
        cv2.putText(frame, "Live Streaming", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    cap.release()