# ### スペースなしでつながっている単語を分割
import wordninja
import re

def separate_words(text):
    """
    This function splits connected words into separate words using WordNinja,
    while preserving punctuation marks as they are.
    """
    # テキストを句読点で分割し、それぞれの部分に処理を適用する
    tokens = re.findall(r'\w+|\S', text)  # 単語と句読点を分離してリスト化
    processed_tokens = []

    for token in tokens:
        if re.match(r'\w+', token):  # 単語部分のみ処理
            processed_token = ' '.join(wordninja.split(token))
            processed_tokens.append(processed_token)
        else:  # 句読点はそのまま保持
            processed_tokens.append(token)

    # 処理されたトークンを結合して文字列を返す
    return ''.join(processed_tokens)


def capitalize(sentence):
    new_sentence = sentence.capitalize()
    # new_sentence = re.sub(r'\bi\b', 'I', new_sentence)
    # new_sentence = re.sub(r'\bi\'', 'I\'', new_sentence)

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