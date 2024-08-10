import numpy as np 
from PIL import Image, ImageDraw
import streamlit as st
import ocr_main

ocr_model = ocr_main.initialize_ocr()

if 'sentences' not in st.session_state:
    st.session_state.sentences = []

# 処理済みファイルを追跡するためのセットを作成
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# アップロードされた画像の表示とOCR処理
image_files = st.file_uploader('画像ファイルを選択してください。', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
if image_files:
    overWrite = st.empty()
    for idx, img in enumerate(image_files):
        # ファイル名を使って、同じファイルが2度処理されないようにする
        if img.name in st.session_state.processed_files:
            continue

        with overWrite.container():
            st.write(f'{idx + 1} / {len(image_files)} 枚目の画像を処理しています・・・。')

        original_image = Image.open(img)
        # OCR処理を行い、セッションステートのsentencesに結果を追加
        st.session_state.sentences += ocr_main.image_to_sentences(np.array(original_image), ocr_model)

        # 処理済みのファイル名を保存
        st.session_state.processed_files.add(img.name)

    overWrite.empty()


sentence_patterns = [
    '第1文型 (SV)',
    '第2文型 (SVC)',
    '第3文型 (SVO)',
    '第4文型 (SVOO)',
    '第5文型 (SVOC)',
    '全て'
]

current_pattern = st.selectbox(
    '文型を選択',
    sentence_patterns,
    index = None,
    placeholder = 'どの文型を取り出しますか'
    )

if current_pattern == '全て':
    st.write(st.session_state.sentences)
