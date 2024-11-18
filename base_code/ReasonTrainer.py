from trainer import MyTrainer
from config.prompts import PROMPT_GEN_REASON_NO_QUESTION_PLUS, PROMPT_GEN_REASON_QUESTION_PLUS

import pandas as pd
from transformers import Trainer
from datasets import Dataset
from evaluate import load
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

class ReasonTrainer(MyTrainer):
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """입력으로 들어온 dataset을 이용하여 입력 프롬프트와 그에 대한 정답을 생성합니다.

        Args:
            dataset (Dataset): 입력으로 들어온 dataset. 아래 필드를 포함하고 있어야 합니다.
            (id-식별자, 지문-paragraph, 질문-question, 선지-choices, 보기-question_plus, 정답-answer
             reason-해설)
        
        Returns:
            Dataset: 아래 필드를 포함하고 있는 dataset을 반환합니다.
            (id-식별자, messages-프롬프트를 위한 message, label-정답)
        """ 
        processed_dataset = []
        for i in range(len(dataset)):
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])
            # <보기>가 있을 때
            if dataset[i]["question_plus"]:
                user_message = PROMPT_GEN_REASON_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = PROMPT_GEN_REASON_NO_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    choices=choices_string,
                )
            # chat message 형식으로 변환
            processed_dataset.append(
                {
                    "id": dataset[i]["id"],
                    "messages": [
                        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": f"{dataset[i]['reason']}"}
                    ],
                    "label": dataset[i]['reason'],
                }
            )
            
        return Dataset.from_pandas(pd.DataFrame(processed_dataset))

    def compute_metrics(self, eval_preds):
        metric = load("bleu")  # BLEU 평가 지표
        logits, labels = eval_preds
        predictions = logits.argmax(dim=-1)

        # Flattening predictions and labels, ignoring padding tokens
        predictions = predictions.view(-1).tolist()
        labels = labels.view(-1).tolist()

        # SacreBLEU expects a list of sentences
        decoded_preds = [self.tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        decoded_labels = [[self.tokenizer.decode(l, skip_special_tokens=True)] for l in labels]

        return metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    def get_trainer(self) -> Trainer:
        """Trainer 객체를 생성합니다. 다음 인스턴스 변수들이 존재해야 합니다.
        - model: 훈련시킬 모델
        - train_dataset: 훈련 데이터셋
        - eval_dataset: 검증 데이터셋
        - data_collator: 데이터 콜레이터 함수
        - tokenizer: 토크나이저
        - params: 훈련에 필요한 하이퍼파라미터

        Returns:
            Trainer: Trainer 객체를 반환합니다.
        """
        peft_config = LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        sft_config = SFTConfig(
            do_train=True,
            do_eval=False,
            lr_scheduler_type=self.params["lr_scheduler_type"],
            max_seq_length=self.params["max_seq_length"],
            output_dir=self.params["output_dir"],
            per_device_train_batch_size=self.params["train_batch_size"],
            per_device_eval_batch_size=self.params["eval_batch_size"],
            num_train_epochs=self.params["epoch"],
            learning_rate=float(self.params["learning_rate"]),
            weight_decay=self.params["weight_decay"],
            logging_steps=100,
            save_strategy="epoch",
            eval_strategy="no",
            save_total_limit=2,
            save_only_model=True,
            report_to="none",
            log_level='error'
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            peft_config=peft_config,
            args=sft_config
        )
        
        print("-" * 30)
        print("lr_scheduler_type : {}".format(self.params["lr_scheduler_type"]))
        print("max_seq_length : {}".format(self.params["max_seq_length"]))
        print("train_batch_size : {}".format(self.params["train_batch_size"]))
        print("eval_batch_size : {}".format(self.params["eval_batch_size"]))
        print("epoch : {}".format(self.params["epoch"]))
        print("learning_rate : {}".format(self.params["learning_rate"]))
        print("weight_decay : {}".format(self.params["weight_decay"]))
        print("-" * 30)
        
        return trainer