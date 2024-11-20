import pandas as pd
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab
from tqdm import tqdm
import json

# BM25 데이터 저장
def save_bm25(tokenized_corpus, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(tokenized_corpus, f, ensure_ascii=False, indent=4)
    print(f"BM25 data saved to {file_path}")

# BM25 데이터 로드
def load_bm25(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tokenized_corpus = json.load(f)
    print(f"BM25 data loaded from {file_path}")
    return tokenized_corpus

# Mecab 초기화
mecab = Mecab()

# 1. 데이터 로드
print("Loading preprocessed data...")
data = pd.read_csv('preprocessed_wikipedia.csv')

# 전처리: 결측값 및 비어 있는 텍스트 처리
data['clean_text'] = data['clean_text'].fillna('')  # NaN -> 빈 문자열
data = data[data['clean_text'].str.strip() != '']  # 빈 문자열 제거

# # 문서 리스트 생성
# documents = data['clean_text'].tolist()

# # 문서 정제: 문자열이 아닌 값을 제거
# documents = [doc if isinstance(doc, str) else '' for doc in documents]

# # 2. 문서 토큰화
# print("Tokenizing documents...")
# tokenized_corpus = [mecab.morphs(doc) for doc in tqdm(documents, desc="Tokenizing")]

# # 결과 확인
# print(f"Tokenized {len(tokenized_corpus)} documents successfully!")

# # 저장
# save_bm25(tokenized_corpus, 'bm25_corpus.json')

# 로드
tokenized_corpus = load_bm25('bm25_corpus.json')

# 3. BM25 객체 생성
print("Initializing BM25...")
bm25 = BM25Okapi(tokenized_corpus)

# 4. 검색 함수 정의
def search(query, top_k=5):
    # 검색어 토큰화
    tokenized_query = mecab.morphs(query)
    # BM25 점수 계산
    scores = bm25.get_scores(tokenized_query)
    # 상위 문서의 인덱스 추출
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    # 결과 반환
    results = []
    for idx in top_k_indices:
        results.append({
            "title": data.iloc[idx]["title"],
            "content": data.iloc[idx]["clean_text"],
            "score": scores[idx]
        })
    return results

# 5. 테스트 검색
print("Testing search...")
query = "(가)은/는 의병계열과 애국계몽 운동 계열의 비밀결사가 모여 결성된 조직으로, 총사령 박상진을 중심으로 독립군 양성을 목적으로 하였다. (가)에 대한 설명으로 옳지 않은 것은?"  # 검색어 입력
results = search(query, top_k=10)

# 6. 검색 결과 출력
print("\nTop Results:")
for i, result in enumerate(results, 1):
    print(f"Rank {i}")
    print(f"Title: {result['title']}")
    print(f"Content: {result['content'][:200]}...")  # 내용 일부만 출력
    print(f"Score: {result['score']}")
    print()
