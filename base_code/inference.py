import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, LoraConfig
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils.load import load_dataset, load_config
import config.prompts as config_prompts

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MyInference:
    def __init__(self, data_path, chk_path):
        self.data_path = data_path + "/test.csv"
        self.checkpoint_path = chk_path

    def inference(self):
        dataset = load_dataset(pd.read_csv(self.data_path))
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True,
        )
        test_dataset = []
        for i, row in dataset.iterrows():
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            len_choices = len(row["choices"])

            # <보기>가 있을 때
            if row["question_plus"]:
                user_message = config_prompts.PROMPT_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = config_prompts.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_string,
                )

            if len(dataset[i]["choices"]) == 4:
                user_message = user_message.replace("1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.", "1, 2, 3, 4 중에 하나를 정답으로 고르세요.")

            test_dataset.append(
                {
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                        {"role": "user", "content": user_message},
                    ],
                    "label": row["answer"],
                    "len_choices": len_choices,
                }
            )
        infer_results = []

        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

        model.eval()
        with torch.inference_mode():
            for data in tqdm(test_dataset):
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                outputs = model(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(DEVICE)
                )

                logits = outputs.logits[:, -1].flatten().cpu()

                target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(target_logit_list, dtype=torch.float32)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
                infer_results.append({"id": _id, "answer": predict_value})
        pd.DataFrame(infer_results).to_csv("output.csv", index=False)
        print("Successfully saved the output csv file!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config", help="path where config json is store")
    parser.add_argument("--checkpoint", type=str, default="./output/checkpoint-7485", help="path where checkpoint dir is store")
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = config["settings"]["dataset"]
    myinference = MyInference(data_path, args.checkpoint)
    myinference.inference()
    