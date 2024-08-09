from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.

import ocr_lib




### Functions ####

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


def initialize_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory


def image_to_sentences(image, ocr_model):
    ocr_result = ocr_model.ocr(image, cls=False)

    ### text, box
    bounding_boxes = []
    bounding_texts = []
    line_height_sum = 0
    line_height_average = 0
    line_count = 0

    for res in ocr_result:
        for line in res:
            bounding_box, (text, *_) = line
            bounding_boxes.append(bounding_box)
            bounding_texts.append(text)
            
            height = int(bounding_box[3][1] - bounding_box[0][1])
            line_height_sum += height
            line_count += 1

    line_height_average = int(line_height_sum / line_count)
    # print('1行の高さ: ', line_height_average)
    # print('Boxの数: ', line_count)


    ### 近くのBox(同じ吹き出し)をマージ
    murged_texts = []
    next_lines = []

    check_next_line(bounding_boxes, line_height_average, next_lines)
    # print("=============== Next Lines ====================")
    # print(next_lines)

    first_lines = [num for num in list(range(len(bounding_boxes))) if num not in next_lines]
    first_lines.sort()
    # print("first_lines:", first_lines)
    # print("num_of_murged_text:", len(first_lines))

    for line_no in first_lines:
        new_text = merge_lines(line_no, bounding_texts, next_lines)
        murged_texts.append(new_text)
        # print(new_text)



    ### アルファベットを含まない要素を削除
    import re
    murged_texts = [s for s in murged_texts if re.search('[a-zA-Z]', s)]


    ### 単語の分割
    murged_texts = list(map(ocr_lib.separate_words, murged_texts))

    ### 文章ごとに分割
    sentences = []
    for text in murged_texts:
        sentences += ocr_lib.split_into_sentences(text)

    ### 先頭文字以外を小文字
    sentences = list(map(str.capitalize, sentences))
    # for s in sentences:
    #     print(s)

    return sentences





## draw result

# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('paddle_result.jpg')


