from typing import Type, Union
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from manager import ModelManager

class EXAONEManager(ModelManager):
    def __init__(self, model_config, params):
        super().__init__(model_config, params)
        self.TEMPLATE = "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ '[|system|][|endofturn|]\n' }}{% endif %}{{ '[|' + message['role'] + '|]' + message['content'] }}{% if message['role'] == 'user' %}{{ '\n' }}{% else %}{{ '[|endofturn|]\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '[|assistant|]' }}{% endif %}"
    
    def set_model(self, AutoModel: Union[Type[AutoModelForCausalLM], Type[AutoPeftModelForCausalLM]] = AutoModelForCausalLM):
        self.model = AutoModel.from_pretrained(
            self.model_name_or_checkpoint,
            trust_remote_code=True,
            device_map="auto",
            fp16=True
        )
        
    def set_data_collator(self):
        response_template = "[|assistant|]"
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )