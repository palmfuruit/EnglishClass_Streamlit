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


# 2つの行が続いているかを判定
def is_next_line(box1, box2, line_height):
    # X方向に重なっている  and  Y方向の距離がline_height以内
    if (box2[0][0] < box1[2][0]) and (box2[1][0] > box1[3][0]):
        y_distance = box2[0][1] - box1[3][1]
        if y_distance > -line_height and y_distance < line_height:
            return True
    
    return False

# 同じ吹き出しの次の行をチェック
def check_next_line(boxes, line_height, next_lines):
    for i in range(0, len(boxes), 1):
        for j in range(i+1, len(boxes), 1):
            if(is_next_line(boxes[i], boxes[j], line_height)):
                # 次の行あり
                next_lines.append(j)
                break
        else:
            # 次の行なし
            next_lines.append('-')

# 同じ吹き出しのテキストを結合
def merge_lines(start_line, texts, next_lines):
    idx = start_line
    text = texts[idx]
    while next_lines[idx] != '-':
        idx = next_lines[idx]
        text +=  ' ' + texts[idx]
    # 行を跨いでた単語を結合
    text = text.replace("- ", "")

    return text