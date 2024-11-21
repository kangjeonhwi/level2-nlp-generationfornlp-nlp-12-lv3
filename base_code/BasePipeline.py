import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import random
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset
from utils.load import load_dataset
import config.prompts as config_prompts
from typing import List

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BasePipeline:
    def __init__(self, data_path: str, model_name_or_checkpoint: str, params: dict):
        """BasePipeline 클래스의 생성자입니다.

        Args:
            data_path (str): train, eval, test 데이터셋이 저장된 디렉토리 경로
            model_name_or_checkpoint (str): 호출할 모델의 이름 혹은 체크포인트 경로
            params (dict): 모델 훈련 설정에 필요한 하이퍼파라미터 (LoRA, Trainer 등)
        """
        self.data_path = data_path
        self.model_name_or_checkpoint = model_name_or_checkpoint
        self.params = params
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def formatting_prompts_func(self, example: dict) -> List[dict]:
        """입력으로 들어온 example을 이용하여 입력 프롬프트를 생성합니다.

        Args:
            example (dict): "messages" 필드를 포함해야 합니다.

        Returns:
            dict: 입력 프롬프트를 반환합니다.
        """
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False
                )
            )
        return output_texts
    
    def tokenize(self, element: dict) -> dict:
        """데이터셋의 각 element를 프롬프트로 변환하고 토크나이징합니다.

        Args:
            element (dict): "messages" 필드를 포함해야 합니다.

        Returns:
            dict: 토크나이징된 결과를 반환합니다.
            - input_ids: 토큰화된 input ids
            - attention_mask: 토큰화된 attention mask
        """
        outputs = self.tokenizer(
            self.formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    def preprocess_logits_for_metrics(self, logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [self.tokenizer.vocab["1"], self.tokenizer.vocab["2"], self.tokenizer.vocab["3"], self.tokenizer.vocab["4"], self.tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx] # -2: answer token, -1: eos token
        return logits
    
    def load_dataset(self) -> Dataset:
        """
        훈련용 데이터셋을 로드하여 허깅페이스 Dataset 형태로 반환합니다.
        Returns:
            Dataset: 훈련용 데이터셋
        """
        return Dataset.from_pandas(load_dataset(pd.read_csv(self.data_path + "/train.csv")))
    
    def make_user_messages(self, row: dict) -> str:
        """데이터셋의 각 row를 이용하여 사용자 프롬프트를 생성합니다.

        Args:
            row (dict): "paragraph", "question", "choices", "question_plus" 필드를 포함해야 합니다.

        Returns:
            str: 사용자 프롬프트를 반환합니다.
        """
        choice_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        
        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = config_prompts.PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choice_string,
            )
        # <보기>가 없을 때
        else:
            user_message = config_prompts.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choice_string,
            )
        
        return user_message
    
    def make_chat_message(self, row: dict, user_message: str) -> dict:
        """데이터셋의 각 row와 그 row로부터 얻은 사용자 프롬프트를 이용하여 챗봇 프롬프트를 포함하는 딕셔너리를 생성합니다.

        Args:
            row (dict): "id", "answer" 필드를 포함해야 합니다.
            user_message (str): 사용자 프롬프트 

        Returns:
            dict: 챗봇 프롬프트를 포함한 딕셔너리. Dataset으로 변환됩니다.
        """
        return {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{row['answer']}"}
            ],
            "label": row["answer"],
        }

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """입력으로 들어온 dataset을 이용하여 입력 프롬프트와 그에 대한 정답을 생성합니다.

        Args:
            dataset (Dataset): 입력으로 들어온 dataset. 아래 필드를 포함하고 있어야 합니다.
            (id-식별자, 지문-paragraph, 질문-question, 선지-choices, 보기-question_plus,정답-answer)
        
        Returns:
            Dataset: 아래 필드를 포함하고 있는 dataset을 반환합니다.
            (id-식별자, messages-프롬프트를 위한 message, label-정답)
        """
        processed_dataset = []
        for i in range(len(dataset)):
            user_message = self.make_user_messages(dataset[i])

            if len(dataset[i]["choices"]) == 4:
                user_message = user_message.replace("1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.", "1, 2, 3, 4 중에 하나를 정답으로 고르세요.")

            # chat message 형식으로 변환
            processed_dataset.append(
                self.make_chat_message(dataset[i], user_message)
            )
        
        return Dataset.from_pandas(pd.DataFrame(processed_dataset))
    
    def compute_metrics(self, evaluation_result):
        int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        acc_metric = evaluate.load("accuracy")

        logits, labels = evaluation_result

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn")[0].strip(), labels))
        labels = list(map(lambda x: int_output_map[x], labels))

        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        acc = acc_metric.compute(predictions=predictions, references=labels)
        
        return acc
   
    def set_model(self):
        """모델을 불러옵니다. 모델은 self.model에 할당합니다.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_checkpoint,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto"
        ).to(DEVICE)
        
    def set_tokenizer(self):
        """토크나이저를 불러옵니다. 토크나이저는 self.tokenizer에 할당합니다.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True
        )
        self.tokenizer.chat_template = config_prompts.TEMPLATE
    
    def set_trainer(self) -> Trainer:
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

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
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

    def report_metrics(self, metrics):
        print("-" * 30)
        print("Evaluation Accuracy : {:.4f}".format(metrics["eval_accuracy"]))
        print("-" * 30)

    def train(self):
        # train task
        dataset = self.load_dataset()
        
        if self.model is None:
            self.set_model()
        if self.tokenizer is None:
            self.set_tokenizer()
        
        processed_dataset = self.process_dataset(dataset)

        tokenized_dataset = processed_dataset.map(
            self.tokenize,
            remove_columns=list(processed_dataset.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        # 데이터 분리
        # 데이터 길이 default = 1024
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)  
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

        self.train_dataset = tokenized_dataset['train']
        self.eval_dataset = tokenized_dataset['test']

        response_template = "<start_of_turn>model"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
        
        if self.trainer is None:
            self.set_trainer()
        
        self.trainer.train()
        final_metrics = self.trainer.evaluate()
        self.report_metrics(final_metrics)