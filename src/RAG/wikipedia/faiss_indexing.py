import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

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

# Encoding 배치 크기 설정
batch_size = 256  # GPU 메모리에 따라 크기를 늘릴 수 있음

# Embedding 생성
print("Generating embeddings with batch processing on GPU...")
embeddings = []

# 데이터를 배치로 분리하여 처리
clean_text_list = data['clean_text'].to_list()
for i in tqdm(range(0, len(clean_text_list), batch_size), desc="Batch encoding"):
    batch = clean_text_list[i:i + batch_size]  # 배치 추출
    batch_embeddings = model.encode(
        batch,
        convert_to_tensor=True,
        device='cuda'  # GPU 사용
    ).cpu().numpy()
    embeddings.append(batch_embeddings)

# Numpy 배열로 변환
embeddings = np.vstack(embeddings)
print("Embedding generation complete!")

# 벡터의 차원 확인
vector_dim = embeddings.shape[1]
print(f"Vector dimension: {vector_dim}")

# FAISS GPU 인덱스 설정
print("Initializing FAISS index with GPU...")
res = faiss.StandardGpuResources()  # GPU 리소스 생성
vector_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dim)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

# 인덱스에 벡터 추가
print("Adding vectors to FAISS index on GPU...")
for embedding in tqdm(embeddings, desc="Adding vectors"):
    gpu_index.add(embedding.reshape(1, -1))

# 저장된 벡터 수 확인
print(f"Total vectors in index: {gpu_index.ntotal}")

# 인덱스 저장
print("Saving FAISS index to file...")
faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), "wikipedia_faiss_index")
print("Index saved as 'wikipedia_faiss_index'.")
