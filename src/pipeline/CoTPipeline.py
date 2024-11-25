import numpy as np
from peft import AutoPeftModelForCausalLM
from pandas.core.api import DataFrame as DataFrame
from pipeline import GenPipeline

class CoTPipeline(GenPipeline):
    def make_chat_message(self, row: dict, user_message: str) -> dict:
        return {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "label": f"{row['reason']}\n\nTherefore, the following choice is the correct answer: {row['answer']}", 
        }
    
    def simple_parse(self, pred: str) -> int:
        x = 0
        for i in range(len(pred), 0, -1):
            if pred[i-1] in ["1", "2", "3", "4", "5"]:
                x = int(pred[i-1])
                break
        return x
    
    def compute_metrics(self, eval_preds):
        metrics = super().compute_metrics(eval_preds) # get BLEU
        
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
        
        parsed_preds = [self.simple_parse(pred) for pred in decoded_preds]
        parsed_labels = [self.simple_parse(label) for label in decoded_labels]
        accuracy = sum([1 for pred, label in zip(parsed_preds, parsed_labels) if pred == label]) / len(parsed_preds)
        metrics["eval_accuracy"] = accuracy
        
        return metrics

    def do_inference(self, model: AutoPeftModelForCausalLM, dataset: DataFrame) -> DataFrame:
        output = super().do_inference(model, dataset)
        output["answer"] = output["reason"].apply(self.simple_parse)
        return output