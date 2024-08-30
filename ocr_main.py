from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.

import ocr_lib


### Functions ####
def initialize_ocr():
    model_dir='./stanza_resources'
    ocr = PaddleOCR(use_angle_cls=True,
                    lang='en',
                    det_model_dir='./paddleocr_resources/det_model',
                    cls_model_dir='./paddleocr_resources/cls_model',
                    rec_model_dir='./paddleocr_resources/rec_model',
    )
    return ocr 


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

    ocr_lib.check_next_line(bounding_boxes, line_height_average, next_lines)
    # print("=============== Next Lines ====================")
    # print(next_lines)

    first_lines = [num for num in list(range(len(bounding_boxes))) if num not in next_lines]
    first_lines.sort()
    # print("first_lines:", first_lines)
    # print("num_of_murged_text:", len(first_lines))

    for line_no in first_lines:
        new_text = ocr_lib.merge_lines(line_no, bounding_texts, next_lines)
        murged_texts.append(new_text)
        # print(new_text)


    ### アルファベットを含まない要素を削除
    import re
    murged_texts = [s for s in murged_texts if re.search('[a-zA-Z]', s)]

    ### 単語間のスペースを補完
    bounding_texts = [ocr_lib.separate_words(text) for text in bounding_texts]

    ### 文章ごとに分割
    # sentences = []
    # for text in murged_texts:
    #     sentences += ocr_lib.split_into_sentences(text)

    ## 先頭文字以外を小文字
    murged_texts = list(map(ocr_lib.capitalize, murged_texts))

    return murged_texts







