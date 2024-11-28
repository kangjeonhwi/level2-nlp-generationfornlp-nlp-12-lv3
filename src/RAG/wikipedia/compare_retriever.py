import bm25s
from dense_retriever import DenseRetriever
from konlpy.tag import Mecab
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_retrievers():
    # BM25 로드
    print("Loading BM25 retriever...")
    bm25_retriever = bm25s.BM25.load("wikipedia_index_bm25", load_corpus=True)
    
    # Dense Retriever 로드
    print("Loading Dense retriever...")
    dense_retriever = DenseRetriever.load("wikipedia_index_dense", load_corpus=True)
    
    return bm25_retriever, dense_retriever

def compare_retrievers(query, bm25_retriever, dense_retriever, k=5):
    # BM25 검색
    mecab = Mecab()
    query_tokens = bm25s.tokenize(' '.join(mecab.morphs(query)))
    bm25_results, bm25_scores = bm25_retriever.retrieve(query_tokens, k=k)
    
    # Dense 검색
    dense_results, dense_scores = dense_retriever.retrieve(query, k=k)
    
    print("=== Query ===")
    print(query)
    print("\n=== BM25 Results ===")
    for i in range(len(bm25_results[0])):
        print(f"Score: {float(bm25_scores[0][i]):.4f}")
        print(f"Text: {bm25_results[0][i]['text'][:200]}...")
        print()
        
    print("\n=== Dense Results ===")
    for i in range(len(dense_results[0])):
        print(f"Score: {float(dense_scores[i]):.4f}")
        print(f"Text: {dense_results[0][i]['text'][:200]}...")
        print()

if __name__ == "__main__":
    # Retriever 로드
    bm25_retriever, dense_retriever = load_retrievers()
    
    # 테스트 쿼리
    test_queries = [
        "이 날 소정방이 부총관 김인문 등과 함께 기 벌포에 도착하여 백제 군사와 마주쳤다. …(중략) …소정방이 신라군이 늦게 왔다는 이유로 군문에서 신라 독군 김문영의 목을 베고자 하니, 그가 군사들 앞에 나아가 “황산 전투를 보지도 않고 늦게 온 것을 이유로 우리를 죄 주려 하는구나. 죄도 없이 치욕을 당할 수는 없으니, 결단코 먼저 당나라 군사와 결전을 한 후에 백제를 쳐야겠다.”라고 말하였다.",
        "(가)은/는 의병계열과 애국계몽 운동 계열의 비밀결사가 모여 결성된 조직으로, 총사령 박상진을 중심으로 독립군 양성을 목적으로 하였다.",
        "○장수왕은 남 진 정책의 일환으로 수도를 이곳으로 천도 하였다. ○묘청은 이곳으로 수도를 옮길 것을 주장하였다."
    ]
    
    # 비교 실행
    for query in test_queries:
        compare_retrievers(query, bm25_retriever, dense_retriever)
        print("\n" + "="*80 + "\n")