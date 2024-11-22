from typing import Type, Union
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from manager import ModelManager

class GemmaManager(ModelManager):
    
    def set_model(self, AutoModel: Union[Type[AutoModelForCausalLM], Type[AutoPeftModelForCausalLM]] = AutoModelForCausalLM):
        self.model = AutoModel.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True,
            device_map="auto"
        )
    