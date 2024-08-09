import numpy as np 
from PIL import Image, ImageDraw
import streamlit as st
import ocr_main

ocr_model = ocr_main.initialize_ocr()
image_files = st.file_uploader('upload image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
sentences = []


# ocr初期化
# ocr_main.ocr_initialize()

# アップロードされた画像を表示
if image_files is None:
    st.write('画像ファイルを選択してください。')

overWrite = st.empty()
for idx, img in enumerate(image_files):
    with overWrite.container():
        st.write(f'{idx + 1} / {len(image_files)} 枚目を処理しています・・・。')

    original_image = Image.open(img)
    # st.image(original_image, caption='Uploaded Image', use_column_width=True)

    sentences += ocr_main.image_to_sentences(np.array(original_image), ocr_model)

st.write(sentences)
