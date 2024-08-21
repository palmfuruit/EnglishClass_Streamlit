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

@st.cache_data
def get_nlp_doc(sentence):
    return st.session_state.nlp(sentence)


def get_span_color(span_type):
    colors = {
        'subject': 'blue',
        'verb': 'red',
        'object': 'yellowgreen',
        'indirect_object': 'green',
        'complement': 'orange',
        'auxiliary': 'pink'
    }
    return colors.get(span_type, 'black')

# 下線スタイルを適用する関数
def apply_underline(text, color):
    return f"<u style='text-decoration-color:{color}; text-decoration-thickness:2pt;'>{text}</u>"



# 主語、動詞、目的語、補語に下線を引く関数
def underline_clauses(text, doc):
    underlined_text = text
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.head == 0:  # ROOT (主節の動詞)
                verb = word.text
                color = get_span_color("verb")
                underlined_text = underlined_text.replace(verb, apply_underline(verb, color), 1)
            elif word.deprel == "nsubj":  # 主語
                subject = word.text
                color = get_span_color("subject")
                underlined_text = underlined_text.replace(subject, apply_underline(subject, color), 1)
            elif word.deprel in ["obj", "iobj"]:  # 目的語
                obj = word.text
                color = get_span_color("object")
                underlined_text = underlined_text.replace(obj, apply_underline(obj, color), 1)
            elif word.deprel in ["xcomp", "ccomp"]:  # 補語
                complement = word.text
                color = get_span_color("complement")
                underlined_text = underlined_text.replace(complement, apply_underline(complement, color), 1)
    return underlined_text


    
def determine_sentence_pattern(spans):
    has_subject = False
    has_object = False
    has_complement = False
    has_object_complement = False
    has_indirect_object = False

    for span, span_type in spans:
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
    print('--------- API Call Start --------')
    if st.session_state.sentences:
        api_url = "http://localhost:8000/predict"
        data = {"text": sentences}
        response = requests.post(api_url, json=data)
        response.raise_for_status()
        response_data = list(response.json())
        # st.write(st.session_state.response_data)
    print('--------- API Call End --------')
    return response_data

@st.cache_data
# 文が該当する文法を表示 (仮定法、比較級、・・・)
def sentence_to_grammer_label(selected_text):
    selected_index = st.session_state.sentences.index(selected_text)
    preds = st.session_state.response_data[selected_index]
    pred_labels = [grammer_labels[idx] for idx, label in enumerate(preds) if label == 1.0]
    pred_labels_html = ""
    for label in pred_labels:
        pred_labels_html += f"<span style='background-color: pink; padding: 2px 4px; margin-right: 5px;'>{label}</span>"

    return pred_labels_html

@st.cache_data
def translate(en_text):
    translated_obj = st.session_state.translator.translate(en_text, dest="ja")
    return translated_obj.text


def get_token_info(doc):
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
    return token_df




@st.cache_data
def display_legend():
    legend_html = """
    <div style='border: 2px solid black; padding: 10px; margin-bottom: 20px;'>
        <p><span style='border-bottom: 2px solid blue; display: inline-block; width: 80px;'></span> 主語(Subject)</p>
        <p><span style='border-bottom: 2px solid red; display: inline-block; width: 80px;'></span> 動詞 (Verb)</p>
        <p><span style='border-bottom: 2px solid pink; display: inline-block; width: 80px;'></span> 助動詞 (Auxiliary)</p>
        <p><span style='border-bottom: 2px solid yellowgreen; display: inline-block; width: 80px;'></span> 目的語 (Object)</p>
        <p><span style='border-bottom: 2px solid green; display: inline-block; width: 80px;'></span> 間接目的語 (Indirect Object)</p>
        <p><span style='border-bottom: 2px solid orange; display: inline-block; width: 80px;'></span> 補語 (Complement)</p>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)



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

    if st.session_state.sentences:
        st.session_state.response_data = predict_grammer_label(st.session_state.sentences)

    # 文の選択
    selected_text = select_text_to_read()
            

    st.divider() # 水平線
    if selected_text:
        # 英文が該当する文法を表示 (仮定法、比較級、・・・)        
        pred_labels_html = sentence_to_grammer_label(selected_text)
        st.write(pred_labels_html, unsafe_allow_html=True)
        

        # # 主語や動詞にアンダーラインを引く
        doc = get_nlp_doc(selected_text)
        main_clause_sentence = underline_clauses(selected_text, doc)
        st.markdown(main_clause_sentence, unsafe_allow_html=True)


        # # 文型
        # spans = extract_spans(doc)
        # sentence_pattern = determine_sentence_pattern(spans)
        # st.write(sentence_pattern)
        

        if st.checkbox("翻訳文を表示"):
            translated_text = translate(selected_text)
            st.write(translated_text)
        
        # トークン情報の表を出力 (開発用)
        token_df = get_token_info(doc)
        st.dataframe(token_df, width=1200)

    # 凡例を表示
    display_legend()

if __name__ == "__main__":
    main()