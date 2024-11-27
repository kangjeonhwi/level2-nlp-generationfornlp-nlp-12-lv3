import os
import torch
from trl import DataCollatorForCompletionOnlyLM
from typing import Type, Union
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from manager import ModelManager

class MistralManager(ModelManager):
    def set_model(self, AutoModel: Union[Type[AutoModelForCausalLM] | Type[AutoPeftModelForCausalLM]] = AutoModelForCausalLM):
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # QLoRA에서 권장하는 nf4 사용
            bnb_4bit_compute_dtype=torch.bfloat16,  # 메모리 효율을 위해 bfloat16 사용
            bnb_4bit_use_double_quant=True,  # QLoRA에서 double quantization 사용
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_checkpoint,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="offload"
        )
        
        for name, param in self.model.named_parameters():
            if "transformer.h.0" in name:  # 예: 초기 레이어를 동결
                param.requires_grad = False
    
    def set_data_collator(self):
        response_template = "[/INST]"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )