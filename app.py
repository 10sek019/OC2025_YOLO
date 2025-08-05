import cv2
import streamlit as st
from ultralytics import YOLO

# ページ構成とモデルの対応
PAGES = {
    "リアルタイム人数カウント": "yolo11n.pt",         # 人検出用（COCO）
    "事前学習モデルで検出": "yolo11n.pt",             # 一般物体検出
    "校章を検出してみよう": "best.pt"     # UOH検出モデル
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

# カメラ初期化関数
def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("カメラが見つかりません")
        return None
    return cap

# モデル読み込み
model = YOLO(model_path)

# 各ページの処理
if page == "リアルタイム人数カウント":
    col1, col2 = st.columns([3, 1])
    img_placeholder = col1.empty()
    count_placeholder = col2.empty()

    cap = get_camera()
    if cap:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb)
            count = 0
            for res in results:
                for box in res.boxes:
                    class_id = int(box.cls)
                    if class_id != 0:
                        continue  # 人以外は無視（COCOのclass 0 = person）

                    x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 255), 5)
                    count += 1

            img_placeholder.image(rgb, use_container_width=True)
            count_placeholder.markdown(f"## 👥 現在の人数: **{count}**")

        cap.release()

else:
    FRAME_WINDOW = st.image([])

    cap = get_camera()
    if cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            flipped = cv2.flip(frame, 1)
            results = model(flipped, verbose=False, conf=0.6 if "校章を検出してみよう" in page else 0.25)

            # ファインチューニングモデルのラベル指定（UOH）
            if "ファインチュ校章を検出してみようーニング" in page:
                results[0].names = {0: "UOH"}

            annotated = results[0].plot()
            FRAME_WINDOW.image(annotated, channels="BGR")

        cap.release()
