from datasets import Dataset
import config.prompts as config_prompts
from trainer import MyTrainer
import pandas as pd
from ast import literal_eval

class CoT2AnswerTrainer(MyTrainer):
    def load_dataset(self) -> Dataset:
        df = pd.read_csv(self.data_path + "/train.csv")
        df = df[df['reason'].notnull()]
        records = []
        for _, row in df.iterrows():
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

if __name__ == "__main__":
    import argparse
    from utils.load import load_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config", help="path where config json is store")
    args = parser.parse_args()
    
    config = load_config(args.config)
    settings = config["settings"]
    trainer = CoT2AnswerTrainer(settings["dataset"], settings["model_name"], config["params"])
    trainer.train()