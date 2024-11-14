import json
from ast import literal_eval
from datasets import Dataset
from box import Box
import pandas as pd

def load_config(config_path: str):
    config_path = config_path + ".json"

    with open(config_path, 'r') as f:
        config = json.load(f)
        config = Box(config)

    return config

def load_dataset(dataset):
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
        }
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)
    df = pd.DataFrame(records)
    return df