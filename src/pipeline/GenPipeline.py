import numpy as np
from ast import literal_eval
from evaluate import load
import pandas as pd
from datasets import Dataset 


from pipeline import BasePipeline 
from prompts import PROMPT_GEN_REASON_QUESTION_PLUS, PROMPT_GEN_REASON_NO_QUESTION_PLUS

class GenPipeline(BasePipeline):
    
    def make_user_messages(self, row: dict) -> str:
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            # <보기>가 있을 때
        if row["question_plus"]:
            user_message = PROMPT_GEN_REASON_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_GEN_REASON_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )
            
        return user_message
    
    def make_chat_message(self, row: dict, user_message: str) -> dict:
        return {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "label": row['reason'],
        }


    # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    def preprocess_logits_for_metrics(self, logits, labels):
        # 로짓에서 가장 높은 확률의 토큰 인덱스를 추출 (argmax)
        predictions = logits.argmax(dim=-1)
        # print(predictions)
        return predictions.cpu(), labels.cpu()

    def compute_metrics(self, eval_preds):
        metric = load("bleu")  # BLEU 평가 지표
        predictions, labels = eval_preds
        predictions = predictions[0] # predictions = Tuple[np.ndarray]
        tokenizer = self.manager.tokenizer
        
        def find_non_pad(row):
            return np.argmax(row != -100)
        answer_indices = np.apply_along_axis(find_non_pad, axis=-1, arr=labels)
        
        for i, idx in enumerate(answer_indices):
            predictions[i, :idx] = -100
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
  
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  
        result = metric.compute(
            predictions=[pred for pred in decoded_preds],  # 토큰화된 예측값
            references=[label for label in decoded_labels]  # 토큰화된 참조값
        )
  
        return result
  
    def load_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in dataset.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
                "reason": row["reason"]
            }
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
        return Dataset.from_pandas(pd.DataFrame(records))
    
    def report_metrics(self, metrics):
        print("-" * 30)
        print("BLEU Score: ", metrics["eval_bleu"])
        print("-" * 30)
