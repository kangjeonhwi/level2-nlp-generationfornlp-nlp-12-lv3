import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from tqdm import tqdm

class CoTDatasetEnhancer:
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature = 0.9,
                 prompt: str = "prompts/enhancer.json",
                 ):
        self._set_secret_key()
        self.prompts = self._load_prompt(prompt)
        self.llm = ChatOpenAI(
            model_name = model_name,
            temperature = temperature
        )
    
    def _load_prompt(self, prompt: str):
        with open(prompt, 'r', encoding="utf-8") as f:
            return json.load(f)
        
    def _set_secret_key(self, secret_file: str = "secrets.json"):
        with open(secret_file, 'r') as f:
            secrets = json.load(f)
        os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
        
    def enhance(self, df):
        system_msg_prompt = SystemMessagePromptTemplate.from_template(
            self.prompts["system"][0]["content"]
        )
        
        parser = JsonOutputParser()
        
        human_msg_prompt = HumanMessagePromptTemplate.from_template(
            self.prompts["user"][0]["content"]
        )
        
        chat_prompt = ChatPromptTemplate(
            messages = [system_msg_prompt, human_msg_prompt],
        )
        
        chat_prompt = chat_prompt.partial(
            format_instance = parser.get_format_instructions()
        )
        
        chain = chat_prompt | self.llm | parser
        
        df = df.copy()
        if "reason" not in df.columns:
            df["reason"] = None
        progress = tqdm(df.iterrows(), total=len(df), desc=f"Enhancing: ")
        total_cost = -1
        for i, row in progress:
            if row["reason"] is not None:
                continue
            paragraph = row["paragraph"]
            problems = json.loads(
                row["problems"].replace("'", "\"")
            )
            question = problems["question"]
            choices = problems["choices"]
            answer = problems["answer"]
            try:
                with get_openai_callback() as cb:
                    jsn = chain.invoke({
                        "paragraph": paragraph,
                        "question": question,
                        "answer": answer,
                        "choices": choices,
                    })
                    total_cost = cb.total_cost
                    progress.set_postfix({
                        "total_cost": f"${total_cost:.2f}",
                    })
            # catch budget issue
            except Exception as _:
                break
            df.loc[i, "reason"] = jsn["reason"]  
        print(f"Total cost: ${total_cost:.2f}")
        
        # if any column of reason is None
        if df["reason"].isnull().sum() > 0:
            print("Some rows are not enhanced! (maybe due to budget issue)")
        
        return df
    
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    
    enhancer = CoTDatasetEnhancer()
    df = enhancer.enhance(df)
    
    df.to_csv("data/train_enhanced.csv", index=False)
    print(df.head())