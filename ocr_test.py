## draw result
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image


ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

image_path = 'D:/python-app/EnglishClassification/img_bleach/Vol20-p050.png'
ocr_result = ocr_model.ocr(image_path, cls=False)

result = ocr_result[0]
image = Image.open(image_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='C:/Windows/Fonts/arial.ttf')
im_show = Image.fromarray(im_show)
im_show.save('paddle_result.jpg')
