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

# 문서를 chunk로 나누는 함수 (1000개의 토큰 크기, 100개 겹침)
def split_into_chunks(text, chunk_size=1000, overlap_size=100):
    words = text.split()  # 공백 기준으로 단어 단위로 분리
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):  # 오버랩 고려하여 분할
        chunk = words[i:i + chunk_size]  # chunk_size만큼 단어 추출
        chunks.append(' '.join(chunk))  # chunk를 텍스트로 결합
        if len(chunk) < chunk_size:  # 마지막 chunk가 chunk_size보다 작으면 종료
            break
    return chunks

# Embedding 생성
print("Generating embeddings with batch processing on GPU...")
embeddings = []
titles = []  # chunk가 속한 원본 문서의 title 저장

# 데이터를 배치로 분리하여 처리
clean_text_list = data['clean_text'].to_list()
for i in tqdm(range(0, len(clean_text_list), batch_size), desc="Batch encoding"):
    batch = clean_text_list[i:i + batch_size]  # 배치 추출
    batch_titles = data['title'].iloc[i:i + batch_size].to_list()  # 배치의 title 추출
    batch_chunks = []
    
    # 각 문서를 chunk로 나누기
    for text in batch:
        chunks = split_into_chunks(text, chunk_size=1000, overlap_size=100)  # 문서를 chunk 단위로 나누기
        batch_chunks.extend(chunks)  # chunk를 모두 추가
    
    # 문서 제목을 각 chunk에 할당
    for title in batch_titles:
        titles.extend([title] * len(batch_chunks))  # 각 제목을 chunk 수만큼 반복하여 할당
    
    # 각 chunk를 임베딩
    batch_embeddings = model.encode(
        batch_chunks,
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
faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), "wikipedia_faiss_index_chunk")
print("Index saved as 'wikipedia_faiss_index_chunk'.")