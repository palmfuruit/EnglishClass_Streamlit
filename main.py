import numpy as np 
from PIL import Image, ImageDraw
import streamlit as st
import requests
import ocr_main

ocr_model = ocr_main.initialize_ocr()

if 'sentences' not in st.session_state:
    st.session_state.sentences = []

# 処理済みファイルを追跡するためのセットを作成
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# アップロードされた画像の表示とOCR処理
image_files = st.file_uploader('画像ファイルを選択してください。', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# 未処理のファイルを取り出し
unprocessed_files = [img for img in image_files if img.name not in st.session_state.processed_files]

if unprocessed_files:
    overWrite = st.empty()
    for idx, img in enumerate(unprocessed_files):

        with overWrite.container():
            st.write(f'{idx + 1} / {len(unprocessed_files)} 枚目の画像を処理しています・・・。')

        original_image = Image.open(img)
        # OCR処理を行い、セッションステートのsentencesに結果を追加
        st.session_state.sentences += ocr_main.image_to_sentences(np.array(original_image), ocr_model)

        # 処理済みのファイル名を保存
        st.session_state.processed_files.add(img.name)

    else:
        overWrite.empty()       # 「N枚目の画像を処理しています・・・。」　消去

        # 取得した文章を文型予測した配列を取得
        api_url = "http://localhost:8000/predict"
        data = {"text": st.session_state.sentences}
        response = requests.post(api_url, json=data)
        response.raise_for_status()


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
