import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from collections import Counter

# DictVectorizerを読み込み
vectorizer = joblib.load('./model/vectorizer.pkl')

# LightGBMモデルを読み込み
models = []
for i in range(6):  # ラベルの数だけループ
    model = lgb.Booster(model_file=f'./model/lightgbm_model_label_{i}.txt')
    models.append(model)

# Stanzaの準備（予測時にも必要）
import stanza
# stanza.download('en', verbose=False) # Stanzaの英語モデルをダウンロード
nlp = stanza.Pipeline('en')

def analyze_sentence(sentence):
    """テキストを解析し、Stanzaの解析結果をデータフレームに変換"""
    doc = nlp(sentence)
    data = []
    for sentence in doc.sentences:
        for word in sentence.words:
            data.append({
                'Text': word.text,
                'Lemma': word.lemma,
                'POS': word.upos,
                'Dependency': word.deprel,
                'Head': word.head,
                'XPOS': word.xpos,
                'Feats': word.feats,
                'Deps': word.deps,  
            })
    return pd.DataFrame(data)

def extract_features(analysis_df):
    """
    文の先頭32トークンのtext, head, pos情報を特徴量として抽出する関数
    """
    features = {}
    max_tokens = 32

    # 文の長さが32トークン未満の場合、不足分を補う
    for i in range(max_tokens):
        if i < len(analysis_df):
            # 現在のトークン情報を取得
            token = analysis_df.iloc[i]
            features[f'token_{i}_text'] = token['Text']
            features[f'token_{i}_lemma'] = token['Lemma']
            features[f'token_{i}_head'] = token['Head']
            features[f'token_{i}_pos'] = token['POS']
            features[f'token_{i}_dependency'] = token['Dependency']
            features[f'token_{i}_xpos'] = token['XPOS']
            features[f'token_{i}_feats'] = token['Feats']
            features[f'token_{i}_deps'] = token['Deps']
        else:
            # トークン数が足りない場合は、デフォルト値を設定
            features[f'token_{i}_text'] = "<PAD>"
            features[f'token_{i}_lemma'] = "<PAD>"
            features[f'token_{i}_head'] = -1
            features[f'token_{i}_pos'] = "<PAD>"
            features[f'token_{i}_dependency'] = "<PAD>"
            features[f'token_{i}_xpos'] = "<PAD>"
            features[f'token_{i}_feats'] = "<PAD>"
            features[f'token_{i}_deps'] = "<PAD>"
    
    return features

def predict_labels(sentences):
    features_list = []
    
    for sentence in sentences:
        analysis_df = analyze_sentence(sentence)
        features = extract_features(analysis_df)
        features_list.append(features)

    X = vectorizer.transform(features_list)
    
    # Initialize an empty list to hold the predictions for all sentences
    all_preds = np.zeros((X.shape[0], len(models)))
    
    # Predict in bulk for all sentences and all models
    for i, model in enumerate(models):
        preds = model.predict(X)
        all_preds[:, i] = preds
    
    # Convert predictions to binary labels
    all_preds_labels = (all_preds >= 0.5).astype(int)
    
    # Convert to list of lists, where each sublist contains the labels for a single sentence
    predictions = all_preds_labels.tolist()
    
    return predictions


# # 使用例
# sentences = [
#     "This is the first test sentence.",
#     "The book was read by the entire class.",
#     "She is taller than her brother.",
#     "The teacher made the students write an essay.",
#     "If I were you, I would apologize immediately.",
#     "The movie that we watched yesterday was amazing.",
#     "This is the place where we first met."
# ]
# predicted_labels = predict_labels(sentences, models, vectorizer)

# # 予測結果の表示
# for i, labels in enumerate(predicted_labels):
#     print(f"Sentence {i+1}: {sentences[i]}")
#     print(f"Predicted Labels: {labels}")
#     print()