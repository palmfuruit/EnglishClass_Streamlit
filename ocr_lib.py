### 文章の分割
from textblob import TextBlob

def split_into_sentences(text):
    blob = TextBlob(text)
    sentence_list = blob.sentences
    sentence_list = [str(sentence) for sentence in sentence_list]

    return sentence_list


### スペースなしでつながっている単語を分割
import wordninja
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def separate_words(text):
    # spacyを使ってテキストをトークンに分割
    doc = nlp(text)
    split_tokens = []

    for token in doc:
        if token.is_alpha:  # 単語のみを処理
            split_tokens.extend(wordninja.split(token.text))
        else:
            split_tokens.append(token.text)
    
    cleaned_text = ''
    for i, token in enumerate(split_tokens):
        if i > 0:
            if token in [',', '.', '!', '?', ':', ';']:
                cleaned_text += token
            elif split_tokens[i - 1] == "'" or token.startswith("'"):
                cleaned_text += token
            else:
                cleaned_text += ' ' + token
        else:
            cleaned_text += token

    # ピリオドの後にアルファベットが続く場合にスペースを追加
    # cleaned_text = re.sub(r'(\.)([A-Za-z])', r'\1 \2', cleaned_text)

    return cleaned_text


def capitalize(sentence):
    new_sentence = sentence.capitalize()
    new_sentence = re.sub(r'\bi\b', 'I', new_sentence)
    new_sentence = re.sub(r'\bi\'', 'I\'', new_sentence)

    return new_sentence
