import os
import json
# from konlpy.tag import Mecab
from konlpy.tag import Okt
import pandas as pd
from tqdm import tqdm  # 진행 상황 표시를 위한 라이브러리

# mecab = Mecab()
okt = Okt()

# 추출된 JSON 데이터 로드
def load_wikipedia_data(path):
    data = []
    for root, _, files in os.walk(path):
        for file in tqdm(files, desc="Loading Wikipedia data"):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    data.append({
                        "id": doc["id"],
                        "title": doc["title"],
                        "text": doc["text"]
                    })
    return pd.DataFrame(data)

# 불필요한 문구 제거 및 형태소 분석
def preprocess_text(text):
    # 예제: 특수문자 및 공백 제거
    text = text.replace('\n', ' ').replace('\t', ' ')
    # 형태소 분석
    # tokens = mecab.morphs(text) # mecab 사용시
    tokens = okt.morphs(text) # okt 사용시
    return ' '.join(tokens)

# 데이터 로드
print("Loading Wikipedia data...")
data = load_wikipedia_data('output_dir')

# 전처리 진행 상황 표시
print("Preprocessing text data...")
tqdm.pandas(desc="Processing text")
data['clean_text'] = data['text'].progress_apply(preprocess_text)

# 저장
print("Saving preprocessed data...")
data[['id', 'title', 'clean_text']].to_csv('preprocessed_wikipedia_okt.csv', index=False)
print("Preprocessing complete. File saved as 'preprocessed_wikipedia_okt.csv'.")