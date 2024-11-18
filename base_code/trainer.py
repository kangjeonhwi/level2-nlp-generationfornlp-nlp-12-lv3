import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import random
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset
from utils.load import load_dataset
import config.prompts as config_prompts

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MyTrainer:
    def __init__(self, data_path, model_name, params):
        self.data_path = data_path
        self.model_name = model_name
        self.params = params

    def formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False
                )
            )

        return output_texts
    
    def tokenize(self, element):
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
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

            # <보기>가 있을 때
            if dataset[i]["question_plus"]:
                user_message = config_prompts.PROMPT_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = config_prompts.PROMPT_NO_QUESTION_PLUS.format(
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
                        {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                    ],
                    "label": dataset[i]["answer"],
                }
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

    def train(self):
        # train task
        dataset = Dataset.from_pandas(load_dataset(pd.read_csv(self.data_path + "/train.csv")))

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto"
        ).to(DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        tokenizer.chat_template = config_prompts.TEMPLATE
        
        processed_dataset = self.process_dataset(dataset)

        self.tokenizer = tokenizer
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
        
        trainer.train()
        final_metrics = trainer.evaluate()

        print("-" * 30)
        print("Evaluation Accuracy : {:.4f}".format(final_metrics["eval_accuracy"]))
        print("-" * 30)