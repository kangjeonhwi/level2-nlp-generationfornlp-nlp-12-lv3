import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 데이터 로드
data = pd.read_csv('preprocessed_wikipedia.csv')

# 데이터 전처리: 비어 있는 값 처리
data['clean_text'] = data['clean_text'].fillna('')  # NaN 값을 빈 문자열로 대체
data = data[data['clean_text'].str.strip() != '']  # 빈 문자열 제거
print(f"Filtered data size: {len(data)} rows")

# 모델 로드 및 GPU 이동
print("Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model.to('cuda')  # GPU로 이동

# CPU에서 인덱스 로드
loaded_index = faiss.read_index("wikipedia_faiss_index")
print("Index loaded from 'wikipedia_faiss_index'.")

# GPU 리소스 생성
res = faiss.StandardGpuResources()

# CPU에서 인덱스 로드 후 GPU로 전환
gpu_index = faiss.index_cpu_to_gpu(res, 0, loaded_index)
print("Index loaded to GPU.")

# 검색 테스트
print("\nTesting FAISS index with a query...")
query = "(가)은/는 의병계열과 애국계몽 운동 계열의 비밀결사가 모여 결성된 조직으로, 총사령 박상진을 중심으로 독립군 양성을 목적으로 하였다."
query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

# 유사한 벡터 검색 (top 5)
k = 5
distances, indices = gpu_index.search(query_embedding, k)

# 결과 출력
print("Top results:")
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}")
    print(f"Title: {data.iloc[idx]['title']}")
    print(f"Content: {data.iloc[idx]['clean_text'][:200]}...")  # 일부 텍스트만 출력
    print(f"Distance: {distances[0][i]}")
    print()