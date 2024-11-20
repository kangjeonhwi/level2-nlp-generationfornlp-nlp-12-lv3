import pandas as pd
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab
from tqdm import tqdm
import pickle

# Mecab 초기화
mecab = Mecab()

# tqdm 초기화
tqdm.pandas(desc="Tokenizing")

# 데이터 로드
print("Loading preprocessed Wikipedia data...")
data = pd.read_csv("preprocessed_wikipedia.csv")

# 전처리: 결측값 처리 및 빈 텍스트 제거
data['clean_text'] = data['clean_text'].fillna('')
data = data[data['clean_text'].str.strip() != '']

# 토큰화
print("Tokenizing documents with Mecab...")
data['tokens'] = data['clean_text'].progress_apply(lambda x: mecab.morphs(x))

# BM25 인덱스 생성
print("Creating BM25 index...")
bm25 = BM25Okapi(data['tokens'].tolist())

# BM25 인덱스 저장
print("Saving BM25 index...")
with open("bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, data), f)
print("BM25 index saved as 'bm25_index.pkl'.")

# 저장된 BM25 인덱스 로드
print("Loading BM25 index and document data...")
with open("bm25_index.pkl", "rb") as f:
    bm25, data = pickle.load(f)

print("BM25 index and document data successfully loaded.")

# 검색 함수 정의
def search(query, top_k=5):
    query_tokens = mecab.morphs(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        result = data.iloc[idx]
        results.append({
            "id": result['id'],
            "title": result['title'],
            "content": result['clean_text'],
            "score": scores[idx]
        })
    return results

# 테스트 쿼리
print("\nTesting BM25 search...")
query = "(가)은/는 의병계열과 애국계몽 운동 계열의 비밀결사가 모여 결성된 조직으로, 총사령 박상진을 중심으로 독립군 양성을 목적으로 하였다. (가)에 대한 설명으로 옳지 않은 것은?"
results = search(query, top_k=5)

# 검색 결과 출력
print("\nTop results:")
for i, res in enumerate(results):
    print(f"Rank {i + 1}:")
    print(f"  Title: {res['title']}")
    print(f"  Content: {res['content'][:200]}...")  # 결과 텍스트 일부만 출력
    print(f"  Score: {res['score']}")
