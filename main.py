import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import requests
import ocr_main
import spacy

# OCRモデルの初期化
ocr_model = ocr_main.initialize_ocr()

# セッションステートの初期化
def initialize_session_state():
    if 'sentences' not in st.session_state:
        st.session_state.sentences = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

# ファイルのアップロードとOCR処理
def process_uploaded_files(image_files, ocr_model):
    unprocessed_files = [img for img in image_files if img.name not in st.session_state.processed_files]
    if unprocessed_files:
        process_images(unprocessed_files, ocr_model)

# 画像の処理
def process_images(unprocessed_files, ocr_model):
    overWrite = st.empty()
    for idx, img in enumerate(unprocessed_files):
        with overWrite.container():
            st.write(f'{idx + 1} / {len(unprocessed_files)} 枚目の画像を処理しています・・・。')
        original_image = Image.open(img)
        st.session_state.sentences += ocr_main.image_to_sentences(np.array(original_image), ocr_model)
        st.session_state.processed_files.add(img.name)
    overWrite.empty()

# SpaCyのセットアップと文の解析
def setup_spacy():
    return spacy.load('en_core_web_sm')

def get_subtree_span(token):
    subtree_tokens = list(token.subtree)
    return subtree_tokens[0].idx, subtree_tokens[-1].idx + len(subtree_tokens[-1].text)

def underline_clauses(sentence, nlp):
    doc = nlp(sentence)
    spans = extract_spans(doc)
    overlap = check_for_overlap(spans)
    
    if overlap:
        main_clause_sentence = apply_annotations(sentence, spans, 'main')
        subordinate_clause_sentence = apply_annotations(sentence, spans, 'subordinate')
        return main_clause_sentence, subordinate_clause_sentence, doc
    else:
        combined_sentence = apply_annotations(sentence, spans)
        return combined_sentence, None, doc

def check_for_overlap(spans):
    # 重なるアンダーラインがあるかチェック
    main_spans = [span for span in spans if span[2] == 'main']
    subordinate_spans = [span for span in spans if span[2] == 'subordinate']
    
    for main_span in main_spans:
        for sub_span in subordinate_spans:
            if main_span[0][0] < sub_span[0][1] and sub_span[0][0] < main_span[0][1]:
                return True
    return False

def extract_spans(doc):
    spans = []
    for token in doc:
        span, span_type, clause_type = get_span_info(token)
        if span:
            spans.append((span, span_type, clause_type))
    return spans

def get_span_info(token):
    if token.pos_ in ['VERB', 'AUX']:
        return (token.idx, token.idx + len(token.text)), 'verb' if token.pos_ == 'VERB' else 'auxiliary', 'main' if token.dep_ == 'ROOT' else 'subordinate'
    elif token.dep_ in ['nsubj', 'csubj', 'nsubjpass', 'csubjpass']:
        return get_subtree_span(token), 'subject', 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
    elif token.dep_ in ['obj', 'dobj', 'iobj']:
        return get_subtree_span(token), 'indirect_object' if token.dep_ == 'iobj' else 'direct_object', 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
    elif token.dep_ in ['attr', 'acomp', 'oprd', 'xcomp']:
        return get_subtree_span(token), 'complement', 'main' if token.head.dep_ == 'ROOT' else 'subordinate'
    return None, None, None

def apply_annotations(sentence, spans, clause_type_filter=None):
    offset_map = [0] * len(sentence)
    annotated_sentence = sentence

    for span, span_type, clause_type in sorted(spans, key=lambda x: x[0][0], reverse=True):
        if clause_type_filter is None or clause_type == clause_type_filter:
            color = get_span_color(span_type)
            max_offset = max(offset_map[span[0]:span[1]]) if offset_map[span[0]:span[1]] else 0
            style = f"border-bottom: 2px {'solid' if clause_type == 'main' else 'double'} {color}; padding-bottom: {max_offset}px;"
            annotated_sentence = (
                annotated_sentence[:span[0]] +
                f"<span style='{style}'>{annotated_sentence[span[0]:span[1]]}</span>" +
                annotated_sentence[span[1]:]
            )
            for i in range(span[0], span[1]):
                offset_map[i] += 4
    return annotated_sentence

def get_span_color(span_type):
    colors = {
        'subject': 'blue',
        'verb': 'red',
        'direct_object': 'yellowgreen',
        'indirect_object': 'green',
        'complement': 'orange',
        'auxiliary': 'pink'
    }
    return colors.get(span_type, 'black')

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

# メイン関数
def main():
    initialize_session_state()

    # アップロードされた画像の表示とOCR処理
    image_files = st.file_uploader('画像ファイルを選択してください。', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    process_uploaded_files(image_files, ocr_model)

    # 文の選択
    selected_text = ""
    if st.session_state.sentences:
        with st.sidebar:
            selected_text = st.radio("分析する文章を選択してください。", st.session_state.sentences)

    # SpaCyの英語モデルをロード
    nlp = setup_spacy()

    st.divider() # 水平線
    if selected_text:
        main_clause_sentence, subordinate_clause_sentence, doc = underline_clauses(selected_text, nlp)
        if subordinate_clause_sentence:
            st.write('<主節>')
            st.markdown(main_clause_sentence, unsafe_allow_html=True)
            st.write('<従属節>')
            st.markdown(subordinate_clause_sentence, unsafe_allow_html=True)
        else:
            st.markdown(main_clause_sentence, unsafe_allow_html=True)
        display_token_info(doc)

    # 凡例を表示
    display_legend()

if __name__ == "__main__":
    main()
