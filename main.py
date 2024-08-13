import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
# import ocr_main
import stanza
# import requests
# import nltk
# from nltk.tokenize import sent_tokenize

# OCRモデルの初期化
# ocr_model = ocr_main.initialize_ocr()

# セッションステートの初期化
def initialize_session_state():
    if 'sentences' not in st.session_state:
        st.session_state.sentences = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'nlp' not in st.session_state:
        # Stanzaの英語モデルをロードしてセッションステートに保存
        st.session_state.nlp = setup_stanza()

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

# Stanzaのセットアップと文の解析
def setup_stanza():
    stanza.download('en')  # Stanzaの英語モデルをダウンロード
    return stanza.Pipeline('en')  # パイプラインの初期化

def get_subtree_span(token, sentence):
    start = token.start_char
    end = token.end_char

    # トークンの子供を再帰的に取得し、最も左の開始位置と最も右の終了位置を求める
    for word in sentence.words:
        if word.head == token.id:  # トークンが現在の単語の親である場合
            child_start, child_end = get_subtree_span(word, sentence)
            start = min(start, child_start)
            end = max(end, child_end)

    return start, end


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
    for sentence in doc.sentences:
        for token in sentence.words:
            span, span_type, clause_type = get_span_info(token, sentence)
            if span:
                spans.append((span, span_type, clause_type))
    return spans


def get_span_info(token, sentence):
    head_token = sentence.words[token.head - 1] if token.head > 0 else None
    if token.head == 0 or (token.upos != 'VERB' and head_token and head_token.deprel == 'root' and token.deprel not in ['ccomp', 'xcomp', 'parataxis']):
        clause_type = 'main'
    else:
        clause_type = 'subordinate'

    if token.upos == 'VERB':    # 動詞
        return (token.start_char, token.end_char), 'verb', clause_type
    elif token.upos == 'AUX':    # 助動詞
        return (token.start_char, token.end_char), 'auxiliary', clause_type
    elif token.deprel in ['nsubj', 'csubj', 'nsubj:pass', 'csubj:pass']:    # 主語
        return get_subtree_span(token, sentence), 'subject', clause_type
    elif token.deprel in ['obj']:
        return get_subtree_span(token, sentence), 'direct_object', clause_type    # 目的語
    elif token.deprel == 'iobj':    # 間接目的語
        return get_subtree_span(token, sentence), 'indirect_object', clause_type
    elif token.deprel in ['cop', 'xcomp'] :
        head_token = sentence.words[token.head - 1] if token.head > 0 else None
        clause_type = 'main' if head_token and head_token.deprel == 'root' else 'subordinate'
        return get_subtree_span(token, sentence), 'complement', clause_type
    elif token.deprel == 'root' and token.upos in ['NOUN', 'ADJ']:
        # 補語 パターン2
        return (token.start_char, token.end_char), 'complement', clause_type
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
        "Text": [word.text for sentence in doc.sentences for word in sentence.words],
        "Lemma": [word.lemma for sentence in doc.sentences for word in sentence.words],
        "POS": [word.upos for sentence in doc.sentences for word in sentence.words],
        "Dependency": [word.deprel for sentence in doc.sentences for word in sentence.words],
        "Head": [
            sentence.words[word.head - 1].text if 0 < word.head <= len(sentence.words) else 'ROOT'
            for sentence in doc.sentences for word in sentence.words
        ],
        "Children": [
            [child.text for child in sentence.words if child.head == word.id]
            for sentence in doc.sentences for word in sentence.words
        ],
        "Start": [word.start_char for sentence in doc.sentences for word in sentence.words],
        "End": [word.end_char for sentence in doc.sentences for word in sentence.words]
    }
    token_df = pd.DataFrame(token_data)
    st.dataframe(token_df)


def determine_sentence_pattern(doc):
    has_subject = False
    has_object = False
    has_complement = False
    has_object_complement = False
    has_indirect_object = False

    for sentence in doc.sentences:
        for token in sentence.words:
            if token.deprel in ['nsubj', 'csubj', 'nsubj:pass', 'csubj:pass']:
                has_subject = True
            elif token.deprel in ['obj', 'iobj']: 
                if token.deprel == 'iobj':
                    has_indirect_object = True
                else:
                    has_object = True
            elif token.deprel in ['cop', 'xcomp']:
                has_complement = True
            elif token.deprel in ['oprd']:
                has_object_complement = True

    if has_subject and not has_object and not has_complement:
        return "第1文型 (SV)"
    elif has_subject and has_complement and not has_object:
        return "第2文型 (SVC)"
    elif has_subject and has_object and not has_object_complement and not has_indirect_object:
        return "第3文型 (SVO)"
    elif has_subject and has_object and has_indirect_object:
        return "第4文型 (SVOO)"
    elif has_subject and has_object and has_object_complement:
        return "第5文型 (SVOC)"
    else:
        return ""


# メイン関数
def main():
    initialize_session_state()

    # ラジオボタンで「テキスト」か「画像」を選択
    input_type = st.radio("入力タイプを選択してください:", ("テキスト", "画像"))

    if input_type == "画像":
        # 画像のアップロードとOCR処理
        image_files = st.file_uploader('画像ファイルを選択してください。', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        process_uploaded_files(image_files, ocr_model)

    elif input_type == "テキスト":
        # テキストボックスと解析ボタンを表示
        text_input = st.text_area("テキストを入力してください:", height=300)
        if st.button("解析"):
            # 入力されたテキストをStanzaで文に分割して保持
            doc = st.session_state.nlp(text_input)
            st.session_state.sentences = [sentence.text for sentence in doc.sentences]

    # 文の選択
    selected_text = ""
    if st.session_state.sentences:
        with st.sidebar:
            selected_text = st.radio("分析する文章を選択してください。", st.session_state.sentences)

    st.divider() # 水平線
    if selected_text:
        main_clause_sentence, subordinate_clause_sentence, doc = underline_clauses(selected_text, st.session_state.nlp)

        # 文型を判定して表示
        sentence_pattern = determine_sentence_pattern(doc)
        st.write(sentence_pattern)
        
        if subordinate_clause_sentence:
            st.write('<主節>')
            st.markdown(main_clause_sentence, unsafe_allow_html=True)
            st.write('  ')
            st.write('<従属節>')
            st.markdown(subordinate_clause_sentence, unsafe_allow_html=True)
        else:
            st.markdown(main_clause_sentence, unsafe_allow_html=True)
        display_token_info(doc)

    # 凡例を表示
    display_legend()

if __name__ == "__main__":
    main()