# import os
import torch
from transformers import AutoTokenizer 
from trl import DataCollatorForCompletionOnlyLM
from manager import MistralManager

class T3QManager(MistralManager):
    def set_data_collator(self):
        response_template = "<|im_start|>assistant\n"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )
        
    def set_tokenizer(self):
        """토크나이저를 불러옵니다. 토크나이저는 self.tokenizer에 할당합니다.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True,
            use_fast = True
        )
        if self.TEMPLATE is not None:
            self.tokenizer.chat_template = self.TEMPLATE
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
        