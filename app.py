import streamlit as st
import cv2
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ページ構成とモデルの対応
PAGES = {
    "リアルタイム人数カウント": "yolov8n.pt",
    "事前学習モデルで検出": "yolov8n.pt",
    "校章を検出してみよう": "best.pt"
}

# ページ状態の初期化
if "page" not in st.session_state:
    st.session_state.page = "リアルタイム人数カウント"

# サイドバーでページ選択（ボタン）
st.sidebar.title("📂 ページ選択")
for name in PAGES.keys():
    if st.sidebar.button(name, key=name):
        st.session_state.page = name

# 現在のページ
page = st.session_state.page
model_path = PAGES[page]

# タイトル表示
st.title(f"📷 {page}")

#---

### リアルタイム人数カウント用クラス

class PersonCounter(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO(PAGES["リアルタイム人数カウント"])
        self.last_frame_time = time.time()
        self.target_fps = 5
        self.person_count = 0

    def transform(self, frame):
        current_time = time.time()
        if current_time - self.last_frame_time < 1 / self.target_fps:
            return frame.to_ndarray(format="bgr24")

        self.last_frame_time = current_time

        img = frame.to_ndarray(format="bgr24")
        flipped_img = cv2.flip(img, 1)

        results = self.model(flipped_img)
        
        count = 0
        for res in results:
            for box in res.boxes:
                class_id = int(box.cls)
                if self.model.names[class_id] != 'person':
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                cv2.rectangle(flipped_img, (x1, y1), (x2, y2), (0, 255, 255), 5)
                count += 1
        
        # クラス内のインスタンス変数にカウントを保存
        self.person_count = count
        
        return flipped_img

#---

### 一般物体検出用クラス

class ObjectDetector(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO(PAGES[page])
        self.last_frame_time = time.time()
        self.target_fps = 5
        
        if "校章を検出してみよう" in page:
            self.conf = 0.6
        else:
            self.conf = 0.25

    def transform(self, frame):
        current_time = time.time()
        if current_time - self.last_frame_time < 1 / self.target_fps:
            return None
        self.last_frame_time = current_time
        
        img = frame.to_ndarray(format="bgr24")
        flipped_img = cv2.flip(img, 1)

        results = self.model(flipped_img, verbose=False, conf=self.conf)

        if "校章を検出してみよう" in page:
            results[0].names = {0: "UOH"}

        annotated_img = results[0].plot()
        return annotated_img

#---

# 各ページの実行とUIの描画
if page == "リアルタイム人数カウント":
    ctx = webrtc_streamer(key="person-counter", video_processor_factory=PersonCounter)
    
    # 映像処理が実行されている間、人数表示を更新
    # ctx.video_processorが存在する場合、そのインスタンスから人数を取得
    if ctx.video_processor:
        # 人数表示用のプレースホルダーを作成
        count_placeholder = st.empty()
        while True:
            # 最新の人数を取得して表示
            if ctx.video_processor.person_count is not None:
                count_placeholder.markdown(f"## 👥 現在の人数: **{ctx.video_processor.person_count}**")
            # 頻繁に更新しすぎないように少し待機
            time.sleep(0.1)

else:
    webrtc_streamer(key="object-detector", video_processor_factory=ObjectDetector)