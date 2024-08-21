import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import ocr_main
import stanza
import requests
# import nltk
# from nltk.tokenize import sent_tokenize
from googletrans import Translator

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# セッションステートの初期化
def initialize_session_state():
    if 'sentences' not in st.session_state:
        st.session_state.sentences = []
    
    if 'response_data' not in st.session_state:
        st.session_state.response_data = []

    if 'nlp' not in st.session_state:
        # Stanzaの英語モデルをロードしてセッションステートに保存
        st.session_state.nlp = setup_stanza()
    
    if 'ocr_model' not in st.session_state:
        st.session_state.ocr_model = ocr_main.initialize_ocr()

    if 'translator' not in st.session_state:
        st.session_state.translator = Translator()

    if 'image_files' not in st.session_state:
        st.session_state.image_files = []
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

# Stanzaのセットアップと文の解析
def setup_stanza():
    stanza.download('en')  # Stanzaの英語モデルをダウンロード
    return stanza.Pipeline('en')  # パイプラインの初期化

# 画像ファイルUpload
def on_file_upload():
    if st.session_state.image_files:
        process_image(st.session_state.image_files)

# 画像の処理
def process_image(image_file):
    overWrite = st.empty()
    with overWrite.container():
        st.info('画像からテキストを読み出し中・・・。')
        original_image = Image.open(image_file)
        st.session_state.sentences = ocr_main.image_to_sentences(np.array(original_image), st.session_state.ocr_model)
    overWrite.empty()
    st.session_state.uploaded_image = original_image

# Readingするテキストを選択
def select_text_to_read():
    selected_text = ""
    if st.session_state.sentences:
        with st.sidebar:
            if st.session_state.uploaded_image: 
                st.image(st.session_state.uploaded_image, use_column_width=True)
            
            grammar_labels_with_counts = get_grammar_label_with_counts()
            selected_grammar = st.selectbox('使用している文法でフィルタ', grammar_labels_with_counts, index=None)
            if selected_grammar:
                selected_grammar = selected_grammar.split(' (')[0]

            if selected_grammar == None:
                selected_text = st.radio("文を選択してください。", st.session_state.sentences)
            else:
                filtered_sentences = []
                for i, preds in enumerate(st.session_state.response_data):
                    pred_labels = [grammer_labels[idx] for idx, label in enumerate(preds) if label == 1.0]
                    if selected_grammar in pred_labels:
                        filtered_sentences.append(st.session_state.sentences[i])
                
                if filtered_sentences:
                    # Display only filtered sentences
                    selected_text = st.radio("文を選択してください。", filtered_sentences)
                else:
                    # Show a message if no sentences match the selected grammar
                    st.write("選択された文法に一致する文がありません。")    
    
    return selected_text



def get_subtree_span(token, sentence):
    start = token.start_char
    end = token.end_char

    # トークンの子供(動詞と句読点除く)を探し、その範囲を確認する
    for word in sentence.words:
        if word.head == token.id and (word.upos not in ['VERB', 'PUNCT']) and (word.deprel not in ['appos', 'conj', 'advmod']):  # トークンが現在の単語の親である場合
            # 子トークンの範囲を確認し、現在の範囲と比較して更新する
            start = min(start, word.start_char)
            end = max(end, word.end_char)

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
    if token.head == 0 or (head_token and head_token.deprel == 'root'):
        clause_type = 'main'
    else:
        clause_type = 'subordinate'


    if token.deprel in ['nsubj', 'csubj', 'nsubj:pass', 'csubj:pass']:    # 主語
        return get_subtree_span(token, sentence), 'subject', clause_type
    elif token.deprel in ['obj']:    # 目的語
        return get_subtree_span(token, sentence), 'direct_object', clause_type
    elif token.deprel == 'iobj':    # 間接目的語
        return get_subtree_span(token, sentence), 'indirect_object', clause_type
    elif token.deprel in ['xcomp', 'ccomp']  and head_token and head_token.upos == 'VERB':
        # 'xcomp'/'ccomp'が動詞、または節の中にBe動詞がある。→目的語
        if token.upos == 'VERB' or any((word.deprel == "cop" and word.head == token.id) for word in sentence.words):
            return get_subtree_span(token, sentence), 'direct_object', clause_type
        else:
            return get_subtree_span(token, sentence), 'complement', clause_type
    elif token.deprel == 'root' and token.upos in ['NOUN', 'ADJ']:      # 補語 パターン2
        return (token.start_char, token.end_char), 'complement', clause_type
    elif token.upos == 'VERB':    # 動詞
        if token.head == 0:
            clause_type = 'main'
        else:
            clause_type = 'subordinate'
        return (token.start_char, token.end_char), 'verb', clause_type
    elif token.upos == 'AUX':    # 助動詞
        return (token.start_char, token.end_char), 'auxiliary', clause_type
    
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
        # "Start": [word.start_char for sentence in doc.sentences for word in sentence.words],
        # "End": [word.end_char for sentence in doc.sentences for word in sentence.words]
    }
    token_df = pd.DataFrame(token_data)
    st.dataframe(token_df, width=1200)


def determine_sentence_pattern(spans):
    has_subject = False
    has_object = False
    has_complement = False
    has_object_complement = False
    has_indirect_object = False

    for span, span_type, clause_type in spans:
        if clause_type == 'main':  # Only consider spans from the main clause
            if span_type == 'subject':
                has_subject = True
            elif span_type == 'direct_object':
                has_object = True
            elif span_type == 'indirect_object':
                has_indirect_object = True
            elif span_type == 'complement':
                has_complement = True

    if has_subject and not has_object and not has_complement:
        return "第1文型 (SV)"
    elif has_subject and has_complement and not has_object:
        return "第2文型 (SVC)"
    elif has_subject and has_object and has_indirect_object:
        return "第4文型 (SVOO)"
    elif has_subject and has_object and has_complement:
        return "第5文型 (SVOC)"
    elif has_subject and has_object:
        return "第3文型 (SVO)"
    else:
        return ""


grammer_labels = [
    '受動態',
    '比較',
    '仮定法',
    '使役',
    '関係代名詞',
    '関係副詞'
]

# 文法ラベルにそれぞれの文法に適合している文の数を追加する関数
def get_grammar_label_with_counts():
    label_counts = {label: 0 for label in grammer_labels}
    
    for preds in st.session_state.response_data:
        for idx, label in enumerate(preds):
            if label == 1.0:
                label_counts[grammer_labels[idx]] += 1
    
    labeled_grammar_labels = [f"{label} ({count})" for label, count in label_counts.items()]
    return labeled_grammar_labels

@st.cache_data
def predict_grammer_label(sentences):
    print('--------- predict_grammer_label() Start --------')
    if st.session_state.sentences:
        api_url = "http://localhost:8000/predict"
        data = {"text": sentences}
        response = requests.post(api_url, json=data)
        response.raise_for_status()
        response_data = list(response.json())
        # st.write(st.session_state.response_data)
    print('--------- predict_grammer_label() End --------')
    return response_data

@st.cache_data
def translate(en_text):
    translated_obj = st.session_state.translator.translate(en_text, dest="ja")
    return translated_obj.text



# メイン関数
def main():
    initialize_session_state()

    st.title('英語Reading学習アプリ')

    # ラジオボタンで「テキスト」か「画像」を選択
    input_type = st.radio("英語テキストの入力方法", ("テキスト", "画像"))

    if input_type == "画像":
        # 画像のアップロードとOCR処理
        image_files = st.file_uploader('英語のテキストが記載された画像を選択', type=['jpg', 'jpeg', 'png'],
                                        accept_multiple_files=False, on_change=on_file_upload, key='image_files')
                                        
    elif input_type == "テキスト":
        # テキストボックスと解析ボタンを表示
        text_input = st.text_area("英語のテキストを入力してください:", height=300)
        if st.button("入力"):
            # 入力されたテキストをStanzaで文に分割して保持
            st.session_state.uploaded_image = None      # 前回アップロードした画像をクリア
            doc = st.session_state.nlp(text_input)
            st.session_state.sentences = [sentence.text for sentence in doc.sentences]


    st.session_state.response_data = predict_grammer_label(st.session_state.sentences)

    # 文の選択
    selected_text = select_text_to_read()
            

    st.divider() # 水平線
    if selected_text:
        # 該当する文法を表示 (仮定法、比較級、・・・)
        selected_index = st.session_state.sentences.index(selected_text)
        preds = st.session_state.response_data[selected_index]
        pred_labels = [grammer_labels[idx] for idx, label in enumerate(preds) if label == 1.0]
        pred_labels_html = ""
        for label in pred_labels:
            pred_labels_html += f"<span style='background-color: pink; padding: 2px 4px; margin-right: 5px;'>{label}</span>"

        st.write(pred_labels_html, unsafe_allow_html=True)
        

        # # 主語や動詞にアンダーラインを引く
        # main_clause_sentence, subordinate_clause_sentence, doc = underline_clauses(selected_text, st.session_state.nlp)

        # # 文型
        # spans = extract_spans(doc)
        # sentence_pattern = determine_sentence_pattern(spans)
        # st.write(sentence_pattern)


        # if subordinate_clause_sentence:
        #     st.write('<主節>')
        #     st.markdown(main_clause_sentence, unsafe_allow_html=True)
        #     st.write('  ')
        #     st.write('<従属節>')
        #     st.markdown(subordinate_clause_sentence, unsafe_allow_html=True)
        # else:
        #     st.markdown(main_clause_sentence, unsafe_allow_html=True)
        
        # # トークン情報の表を出力 (開発用)
        # display_token_info(doc)
        
        st.write(selected_text)

        if st.checkbox("翻訳文を表示"):
            translated_text = translate(selected_text)
            st.write(translated_text)
        

    # 凡例を表示
    display_legend()

if __name__ == "__main__":
    main()