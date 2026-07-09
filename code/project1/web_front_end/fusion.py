# fusion.py
import os
import cv2
import torch
import numpy as np
import urllib.request
from PIL import Image
from torchvision import transforms
from models import EnhancedOfflineBERT, EmotionResNet
from tokenizer import EnhancedTokenizer
from config import YU_NET_MODEL_PATH

class YuNetFaceDetector:
    def __init__(self, model_path=YU_NET_MODEL_PATH, conf_threshold=0.6, nms_threshold=0.3, top_k=5000):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = None
        self._init_detector()
        self.last_faces = []
        self.last_frame_id = -1

    def _download_model(self):
        if os.path.exists(self.model_path):
            return True
        url = "https://github.com/opencv/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        print("正在下载 YuNet 模型...")
        try:
            urllib.request.urlretrieve(url, self.model_path)
            print("模型下载完成")
            return True
        except Exception as e:
            print(f"模型下载失败: {e}")
            return False

    def _init_detector(self):
        if not self._download_model():
            print("⚠️ YuNet 模型不可用，将回退到 Haar 级联")
            self.detector = None
            return
        self.detector = cv2.FaceDetectorYN.create(
            self.model_path, "", (320, 320),
            self.conf_threshold, self.nms_threshold, self.top_k
        )
        if self.detector is None:
            print("⚠️ YuNet 初始化失败，将回退到 Haar 级联")
        else:
            print("✅ YuNet 人脸检测器加载成功")

    def detect_faces(self, frame, roi=None, force=False, frame_id=0):
        if not force and frame_id == self.last_frame_id:
            return self.last_faces
        self.last_frame_id = frame_id
        if self.detector is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
            all_boxes = [(x, y, w, h) for (x, y, w, h) in faces]
        else:
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is None:
                all_boxes = []
            else:
                all_boxes = []
                for face in faces:
                    x, y, wb, hb = face[:4].astype(int)
                    all_boxes.append((x, y, wb, hb))
        if roi is None:
            self.last_faces = all_boxes
            return all_boxes
        rx, ry, rw, rh = roi
        filtered = []
        for (x, y, wb, hb) in all_boxes:
            cx = x + wb // 2
            cy = y + hb // 2
            if rx <= cx <= rx+rw and ry <= cy <= ry+rh:
                filtered.append((x, y, wb, hb))
        self.last_faces = filtered
        return filtered

    def get_largest_face(self, frame, roi=None, force=False, frame_id=0):
        boxes = self.detect_faces(frame, roi, force, frame_id)
        if not boxes:
            return None, None
        x, y, w, h = max(boxes, key=lambda b: b[2]*b[3])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y:y+h, x:x+w]
        return face_roi, (x, y, w, h)


class MultiModalEmotionFusion:
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    TEXT_TO_7 = np.array([
        [0.25, 0.25, 0.25, 0.0,  0.25, 0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.5,  0.0,  0.5,  0.0]
    ])

    def __init__(self, text_model_path, visual_model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 构建词表（此处直接调用完整语料，可以从单独文件导入，为简化直接放这里）
        all_texts = self._get_training_corpus()
        self.tokenizer = EnhancedTokenizer()
        self.tokenizer.build_vocab(all_texts)
        vocab_size = len(self.tokenizer.vocab)
        state_dict = torch.load(text_model_path, map_location='cpu')
        actual_vocab_size = state_dict['embedding.weight'].shape[0]
        if actual_vocab_size != vocab_size:
            print(f"⚠️ 文本模型词表大小不一致：训练={actual_vocab_size}，当前构建={vocab_size}，使用训练时大小")
            vocab_size = actual_vocab_size
        self.text_model = EnhancedOfflineBERT(vocab_size=vocab_size).to(self.device)
        self.text_model.load_state_dict(state_dict)
        self.text_model.eval()
        print(f"✅ 文本模型加载成功 (vocab_size={vocab_size})")
        self.visual_model = EmotionResNet(num_classes=7).to(self.device)
        self.visual_model.load_state_dict(torch.load(visual_model_path, map_location=self.device))
        self.visual_model.eval()
        print("✅ 视觉模型加载成功")
        self.face_detector = YuNetFaceDetector()
        self.vis_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _get_training_corpus(self):
        # 这里放置完整语料（positive+negative+neutral），可从原文件复制
        # 为了节省篇幅，此处省略，实际请从之前完整代码中复制
        positive = ["主播太厉害！", "666", "太强了", "操作好丝滑", "牛逼", "主播加油",
        "太秀了", "爱了爱了", "好厉害", "这操作绝了", "支持主播", "稳！",
        "666666", "学到了", "太强了吧", "操作拉满", "太强了太强了", "爱了",
        "主播yyds", "牛啊牛啊", "厉害厉害", "太强了太强了", "丝滑！", "舒服",
        "主播太厉害了","666","太强了","操作好丝滑","牛逼","主播加油",
        "太秀了","爱了爱了","好厉害","这操作绝了","支持主播","稳！",
        "666666","学到了","太强了吧","操作拉满","太强了太强了","爱了",
        "主播yyds","牛啊牛啊","厉害厉害","丝滑！","舒服","看得过瘾",
        "这波操作满分","大神请收下膝盖","全程高能","太精彩了","实力主播",
        "全程目不转睛","越看越上头","技术一流","状态拉满","完美发挥",
        "佩服佩服","简直无敌","看点满满","继续保持","未来可期",
        "操作行云流水","大佬就是大佬","名场面预定","太强了我的哥",
        "全程无失误","看得很爽","实力在线","圈粉了","再来一波",
        "精彩精彩","表现亮眼","发挥稳定","值得一看",
        "讲得特别清楚，学到很多东西","内容干货满满，太实用了","主播口才真好，听得很投入",
        "推荐的东西性价比很高，已经下单了","唱歌好好听，循环听了好几遍","讲解细致又耐心，必须点赞",
        "观点很独到，刷新了我的认知","直播间氛围特别好，越待越开心","分享的经验很受用，感谢分享",
        "画面画质清晰，观看体验很棒","这个讲解逻辑清晰，一听就懂","人美歌甜，太喜欢了",
        "知识讲解通俗易懂，新手也能学会","好物推荐很用心，没有乱推销","互动很有趣，一点都不无聊",
        "分析得有理有据，非常佩服","终于等到直播了，期待好久","做事认真负责，值得信任",
        "文案写得真好，很有感染力","旅行风景太美了，看得心旷神怡","科普内容做得很棒，涨知识了",
        "穿搭风格超好看，跟着学搭配了","美食看起来太有食欲了","手工做得太精致了，手艺一流",
        "观点三观很正，说得很有道理","节奏把握得很好，全程不拖沓","分享的生活小技巧太实用了",
        "声音温柔好听，听着特别治愈","测评真实客观，参考价值很高","舞台表现力超强，气场十足",
        "内容质量很高，每天都会来打卡","待人友善亲切，像朋友聊天一样","科普内容有趣不枯燥",
        "家居布置得好温馨，借鉴一下","健身动作讲解标准，跟着练没问题","朗读感情饱满，特别有感染力",
        "测评很细致，方方面面都讲到了","户外风景绝美，看得心情舒畅","育儿经验很实用，收藏了",
        "设计创意十足，太有想法了","说话幽默风趣，笑点不断","干货超多，笔记记不停",
        "服务态度很好，有问必答","画作功底深厚，艺术感拉满","探店内容真实，种草成功"]   # 请复制完整列表
        negative = ["什么垃圾操作", "太菜了吧", "会不会玩", "辣鸡", "这都能输", "能不能认真点",
        "服了", "绝了真的菜", "下饭操作", "不想看了", "你好菜呀", "垃圾",
        "这水平也敢播", "太拉胯了", "下饭", "真的菜", "什么玩意", "菜死了",
        "太垃圾了吧", "菜鸡", "这操作我上我也行", "吐了", "无语了",
        "什么垃圾操作","太菜了吧","会不会玩","辣鸡","这都能输","能不能认真点",
        "服了","绝了真的菜","下饭操作","不想看了","你好菜呀","垃圾",
        "这水平也敢播","太拉胯了","下饭","真的菜","什么玩意","菜死了",
        "太垃圾了吧","菜鸡","这操作我上我也行","吐了","无语了","离谱到家",
        "全程失误不断","看得着急","浪费时间","水平一般还装逼","心态崩了",
        "越打越菜","毫无看点","失望透顶","就这水平？","太差劲了",
        "操作僵硬","频频翻车","根本不会玩","纯纯混子","越看越生气",
        "技术一言难尽","失误太多了","完全不在状态","尬住了","劝退",
        "毫无技术可言","操作稀烂","心态炸了","看得难受","不想继续看",
        "内容杂乱无章，完全听不懂","推荐的东西质量很差，不建议购买","说话吞吞吐吐，一点都不流畅",
        "全程敷衍了事，根本没用心讲解","声音太难听了，听得很不舒服","内容水时长，没一点干货",
        "夸大宣传，实物和描述差距太大","态度傲慢，提问也不回复","讲解漏洞百出，误导观众",
        "画面模糊卡顿，观看体验极差","强行带货，观感很不舒服","观点偏激，完全无法认同",
        "唱歌跑调严重，实在听不下去","全程念稿子，毫无个人想法","故意制造噱头，博眼球而已",
        "讲解潦草，关键内容一笔带过","商品价格虚高，完全不值这个价","语气阴阳怪气，让人反感",
        "内容重复来回说，越看越无聊","测评弄虚作假，刻意美化产品","直播全程冷场，气氛尴尬",
        "讲解专业度不足，很多常识都搞错了","一味炒作话题，没有实质内容","穿搭造型一言难尽",
        "美食看着毫无食欲，卖相很差","手工做工粗糙，细节处理不到位","科普内容错误百出，误人子弟",
        "说话啰嗦拖沓，半天讲不到重点","户外拍摄杂乱，毫无美感可言","恶意抬价，性价比极低",
        "互动态度恶劣，随意怼观众","作品毫无创意，全是模仿别人","健身动作讲解错误，容易受伤",
        "朗读毫无感情，像机器人一样","探店夸大其词，实际体验很差","家居搭配杂乱，毫无美感",
        "育儿建议不科学，千万别照搬","舞台表现僵硬，毫无感染力","满口空话，没有实际内容",
        "回答问题敷衍，随便应付两句","作品粗制滥造，看不出用心之处","频繁插广告，影响观看","艹","NM","CNM"]
        neutral = ["还行吧", "一般般", "就这样", "还行", "不算好也不算差", "中规中矩",
        "勉强可以", "一般", "还行还行", "就那样吧", "哦", "知道了",
        "没什么感觉", "一般般吧", "就那样", "还行还行", "哦豁", "正常水平",
        "还行吧","一般般","就这样","还行","不算好也不算差","中规中矩",
        "勉强可以","一般","还行还行","就那样吧","哦","知道了",
        "没什么感觉","一般般吧","就那样","还行还行","哦豁","正常水平",
        "普通发挥","不好不坏","中规中矩吧","也就这样","平淡无奇",
        "无功无过","日常操作","没亮点也没槽点","正常发挥","平平淡淡",
        "常规操作","也就一般水平","没啥特别的","普通表现","看得过去",
        "不吹不黑","客观来说一般","日常直播状态","没啥感觉","路过看看",
        "随便看看","正常水准","普通操作而已","整体一般","不好不坏吧",
        "中规中矩的发挥","没啥惊喜","保持现状","常规表现","正常发挥罢了",
        "内容中规中矩，没有太多亮点","就正常聊聊日常，打发时间而已","价格中等，不算贵也不算便宜",
        "声音普通，没有特别的特色","讲解流程常规，和其他主播差不多","画面清晰度一般，够用就行",
        "观点比较大众化，大家基本都这么想","歌曲唱得还行，不好也不坏","分享的内容见过不少了",
        "直播间人数不多，安安静静的","简单介绍一下产品，常规流程","风格平平淡淡，没什么记忆点",
        "日常闲聊，随便看看就走","讲解速度适中，不快也不慢","实物和图片差距不大，正常水平",
        "内容篇幅不长，很快就结束了","穿搭风格比较大众，属于普通款式","美食口味常规，家常味道",
        "手工样式普通，日常款式而已","科普内容都是基础常识","旅行景色一般，普通风景",
        "健身动作都是基础入门动作","朗读水平中等，无功无过","探店场所就是普通门店",
        "家居布局比较传统，没新意","说话语气平和，普普通通","创意比较常规，见过类似的作品",
        "互动不多，各看各的","知识难度适中，大众都能理解","直播时长正常，不长不短",
        "商品款式常见，市面上很多同款","节奏平稳，没有起伏变化","客观陈述事实，不带偏向",
        "生活分享内容，比较接地气","测评结果中规中矩，优缺点都有","舞台表演流程常规",
        "语速正常，听着不费力","内容题材很常见，不算新颖","整体表现平稳，没有意外亮点",
        "简单分享见闻，随便听听","做工符合常规标准，正常品质","氛围安静，适合休闲观看"]
        return positive + negative + neutral

    def predict_visual(self, frame_bgr, roi=None, force_face=False, frame_id=0):
        face_roi, _ = self.face_detector.get_largest_face(frame_bgr, roi, force=force_face, frame_id=frame_id)
        if face_roi is None:
            return None, 0.0, None
        h,w=face_roi.shape[:2]
        if h==0 or w==0:
            return None,0.0,None
        face_pil = Image.fromarray(face_roi)
        tensor = self.vis_transform(face_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.visual_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        conf = np.max(probs)
        idx = np.argmax(probs)
        return self.EMOTIONS[idx], conf, probs

    def predict_text(self, text):
        if not text or text.strip() == "":
            return None, 0.0
        vec = self.tokenizer.encode(text)
        input_tensor = torch.tensor([vec]).to(self.device)
        with torch.no_grad():
            logits = self.text_model(input_tensor)
            probs_3 = torch.softmax(logits, dim=1)[0].cpu().numpy()
        probs_7 = probs_3 @ self.TEXT_TO_7
        conf = np.max(probs_7)
        return probs_7, conf

    def predict_multimodal(self, text, frame_bgr, roi=None, force_face=False, frame_id=0):
        text_probs_7, text_conf = self.predict_text(text)
        weights = (0, 0)
        if text_probs_7 is None:
            vis_emotion, vis_conf, _ = self.predict_visual(frame_bgr, roi, force_face, frame_id)
            if vis_emotion:
                return {'emotion': vis_emotion, 'confidence': vis_conf, 'text_conf': 0, 'visual_conf': vis_conf, 'weights': (0, vis_conf), 'visual_emotion': vis_emotion}
            else:
                return {'emotion': 'No face', 'confidence': 0, 'text_conf': 0, 'visual_conf': 0, 'weights': (0,0), 'visual_emotion': None}
        vis_emotion, vis_conf, vis_probs = self.predict_visual(frame_bgr, roi, force_face, frame_id)
        if vis_probs is None:
            final_probs = text_probs_7
            final_conf = text_conf
            weights = (text_conf, 0)
            final_emotion = self.EMOTIONS[np.argmax(final_probs)]
        else:
            w_text = text_conf
            w_vis = vis_conf
            final_probs = (w_text * text_probs_7 + w_vis * vis_probs) / (w_text + w_vis)
            final_conf = np.max(final_probs)
            final_emotion = self.EMOTIONS[np.argmax(final_probs)]
            weights = (w_text, w_vis)
        return {
            'emotion': final_emotion,
            'confidence': final_conf,
            'text_conf': text_conf,
            'visual_conf': vis_conf,
            'weights': weights,
            'visual_emotion': vis_emotion
        }

    def draw_faces(self, frame, roi=None, frame_id=0):
        boxes = self.face_detector.detect_faces(frame, roi, force=False, frame_id=frame_id)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame