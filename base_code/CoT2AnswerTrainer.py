from datasets import Dataset
import config.prompts as config_prompts
from trainer import MyTrainer
import pandas as pd

class CoT2AnswerTrainer(MyTrainer):
    def process_dataset(self, dataset: Dataset) -> Dataset:
        processed_dataset = []
        for i in range(len(dataset)):
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

            # <보기>가 있을 때
            if dataset[i]["question_plus"]:
                user_message = config_prompts.PROMPT_W_REASON_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                    reason=dataset[i]["reason"],
                )
            # <보기>가 없을 때
            else:
                user_message = config_prompts.PROMPT_W_REASON_NO_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    choices=choices_string,
                    reason=dataset[i]["reason"],
                )

            if len(dataset[i]["choices"]) == 4:
                user_message = user_message.replace("1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.", "1, 2, 3, 4 중에 하나를 정답으로 고르세요.")

            # chat message 형식으로 변환
            processed_dataset.append(
                {
                    "id": dataset[i]["id"],
                    "messages": [
                        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                    ],
                    "label": dataset[i]["answer"],
                }
            )
        
        return Dataset.from_pandas(pd.DataFrame(processed_dataset))