import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import requests
import ocr_main

ocr_model = ocr_main.initialize_ocr()

if 'sentences' not in st.session_state:
    st.session_state.sentences = []
    st.session_state.sentences1 = []
    st.session_state.sentences2 = []
    st.session_state.sentences3 = []
    st.session_state.sentences4 = []
    st.session_state.sentences5 = []

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

        st.session_state.sentences1 = [st.session_state.sentences[i] for i in range(len(response_data)) if response_data[i]["pattern"] == 1]
        st.session_state.sentences2 = [st.session_state.sentences[i] for i in range(len(response_data)) if response_data[i]["pattern"] == 2]
        st.session_state.sentences3 = [st.session_state.sentences[i] for i in range(len(response_data)) if response_data[i]["pattern"] == 3]
        st.session_state.sentences4 = [st.session_state.sentences[i] for i in range(len(response_data)) if response_data[i]["pattern"] == 4]
        st.session_state.sentences5 = [st.session_state.sentences[i] for i in range(len(response_data)) if response_data[i]["pattern"] == 5]



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

selected_sentences = []
if current_pattern == '全て':
    selected_sentences = st.session_state.sentences
elif current_pattern == sentence_patterns[0]:
    selected_sentences = st.session_state.sentences1
elif current_pattern == sentence_patterns[1]:
    selected_sentences = st.session_state.sentences2
elif current_pattern == sentence_patterns[2]:
    selected_sentences = st.session_state.sentences3
elif current_pattern == sentence_patterns[3]:
    selected_sentences = st.session_state.sentences4
elif current_pattern == sentence_patterns[4]:
    selected_sentences = st.session_state.sentences5

# st.write(selected_sentences)
# ラジオボタンでテキストを選択
selected_text = ""
if selected_sentences:
    selected_text = st.radio("Select text", selected_sentences)



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

    for token in doc:
        if token.dep_ in {'nsubj', 'csubj', 'nsubjpass', 'csubjpass'}:  # 主語
            span = get_subtree_span(token)
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'subject', clause_type))
        elif token.pos_ == 'VERB':  # 動詞
            span = (token.idx, token.idx + len(token.text))
            clause_type = 'main' if token.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'verb', clause_type))
        elif token.dep_ in {'dobj', 'iobj'}:  # 目的語
            span = get_subtree_span(token)
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            if token.dep_ == 'dobj':
                spans.append((span, 'direct_object', clause_type))  # 直接目的語
            else:
                spans.append((span, 'indirect_object', clause_type))  # 間接目的語
        elif token.dep_ in {'attr', 'acomp', 'oprd'}:  # 補語
            span = get_subtree_span(token)
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'complement', clause_type))
        elif token.dep_ == 'aux':  # 助動詞
            span = (token.idx, token.idx + len(token.text))
            clause_type = 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
            spans.append((span, 'auxiliary', clause_type))

    # HTMLとCSSを使って下線と色を追加
    annotated_sentence = sentence
    for span, span_type, clause_type in sorted(spans, key=lambda x: x[0][0], reverse=True):
        if span_type == 'subject':
            color = 'blue'
        elif span_type == 'verb':
            color = 'red'
        elif span_type == 'direct_object':
            color = 'green'
        elif span_type == 'indirect_object':
            color = 'yellowgreen'
        elif span_type == 'complement':
            color = 'orange'
        elif span_type == 'auxiliary':  # 助動詞
            color = 'pink'

        if clause_type == 'main':
            style = f"border-bottom: 2px solid {color};"
        else:  # 'subordinate'
            style = f"border-bottom: 2px double {color};"

        annotated_sentence = (
            annotated_sentence[:span[0]] + 
            f"<span style='{style}'>{annotated_sentence[span[0]:span[1]]}</span>" + 
            annotated_sentence[span[1]:]
        )
    
    return annotated_sentence, doc



def display_legend():
    legend_html = """
    <div style='border: 2px solid black; padding: 10px; margin-bottom: 20px;'>
        <p><span style='color: blue;'>■</span> 主語 (Subject)</p>
        <p><span style='color: red;'>■</span> 動詞 (Verb)</p>
        <p><span style='color: green;'>■</span> 直接目的語 (Direct Object)</p>
        <p><span style='color: yellowgreen;'>■</span> 間接目的語 (Indirect Object)</p>
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