import numpy as np 
import pandas as pd
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
        response_data = response.json()
        # st.write(response_data)

 


# st.write(st.session_state.sentences)
# ラジオボタンでテキストを選択
selected_text = ""
if st.session_state.sentences:
    with st.sidebar:
        selected_text = st.radio("分析する文章を選択してください。", st.session_state.sentences)



import spacy

# SpaCyの英語モデルをロード
nlp = spacy.load('en_core_web_sm')


def get_subtree_span(token):
    # トークンのサブツリー（そのトークンを含むすべての子孫ノード）の範囲を取得
    subtree_tokens = list(token.subtree)
    return subtree_tokens[0].idx, subtree_tokens[-1].idx + len(subtree_tokens[-1].text)

def underline_clauses(sentence):
    # 文を解析
    doc = nlp(sentence)
    
    spans = []
    offset_map = [0] * len(sentence)  # 文字ごとのずらし量を格納するリスト

    for token in doc:
        if token.pos_ == 'VERB':  # 動詞
            span = (token.idx, token.idx + len(token.text))
            clause_type = 'main' if token.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'verb', clause_type))
        elif token.pos_ == 'AUX':  # 助動詞
            span = (token.idx, token.idx + len(token.text))
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'auxiliary', clause_type))
        
        elif token.dep_ in {'nsubj', 'csubj', 'nsubjpass', 'csubjpass'}:  # 主語
            span = get_subtree_span(token)
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'subject', clause_type))
        elif token.dep_ in {'obj', 'dobj', 'iobj'}:  # 目的語
            span = get_subtree_span(token)
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            if token.dep_ == 'iobj':
                spans.append((span, 'indirect_object', clause_type))  # 間接目的語
            else:
                spans.append((span, 'direct_object', clause_type))  # 直接目的語
        elif token.dep_ in {'attr', 'acomp', 'oprd', 'xcomp'}:  # 補語
            span = get_subtree_span(token)
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'complement', clause_type))

    # HTMLとCSSを使って下線と色を追加
    annotated_sentence = sentence
    for span, span_type, clause_type in sorted(spans, key=lambda x: x[0][0], reverse=True):
        if span_type == 'subject':
            color = 'blue'
        elif span_type == 'verb':
            color = 'red'
        elif span_type == 'direct_object':
            color = 'yellowgreen'
        elif span_type == 'indirect_object':
            color = 'green'
        elif span_type == 'complement':
            color = 'orange'
        elif span_type == 'auxiliary':  # 助動詞
            color = 'pink'

        # このスパンが重なっている部分の最大オフセットを計算
        max_offset = max(offset_map[span[0]:span[1]]) if offset_map[span[0]:span[1]] else 0

        if clause_type == 'main':
            style = f"border-bottom: 2px solid {color}; padding-bottom: {max_offset}px;"
        else:  # 'subordinate'
            style = f"border-bottom: 2px double {color}; padding-bottom: {max_offset}px;"

        # アンダーラインを追加
        annotated_sentence = (
            annotated_sentence[:span[0]] + 
            f"<span style='{style}'>{annotated_sentence[span[0]:span[1]]}</span>" + 
            annotated_sentence[span[1]:]
        )

        # このスパンの範囲にオフセットを追加
        for i in range(span[0], span[1]):
            offset_map[i] += 4  # 4pxずつオフセットを追加
    
    return annotated_sentence, doc

def display_legend():
    legend_html = """
    <div style='border: 2px solid black; padding: 10px; margin-bottom: 20px;'>
        <p><span style='color: blue;'>■</span> 主語 (Subject)</p>
        <p><span style='color: red;'>■</span> 動詞 (Verb)</p>
        <p><span style='color: yellowgreen;'>■</span> 目的語 (Object)</p>
        <p><span style='color: green;'>■</span> 間接目的語 (Indirect Object)</p>
        <p><span style='color: orange;'>■</span> 補語 (Complement)</p>
        <p><span style='color: pink;'>■</span> 助動詞 (Auxiliary)</p>
        <p><span style='border-bottom: 2px solid black; display: inline-block; width: 80px;'>      </span> 主節 (Main Clause)</p>
        <p><span style='border-bottom: 2px double black; display: inline-block; width: 80px;'>      </span> 従属節 (Subordinate Clause)</p>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


def display_token_info(doc):
    token_data = {
        "Text": [token.text for token in doc],
        "Lemma": [token.lemma_ for token in doc],
        "POS": [token.pos_ for token in doc],
        "Tag": [token.tag_ for token in doc],
        "Dependency": [token.dep_ for token in doc],
        "Head": [token.head.text for token in doc],
        "Children": [[child.text for child in token.children] for token in doc],
        "Start": [token.idx for token in doc],
        "End": [token.idx + len(token.text) for token in doc]
    }
    
    token_df = pd.DataFrame(token_data)
    st.dataframe(token_df)


st.divider() # 水平線
if selected_text:
    underlined_text, doc = underline_clauses(selected_text)
    st.markdown(underlined_text, unsafe_allow_html=True)
    display_token_info(doc)


# 凡例を表示
display_legend()