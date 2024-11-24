from pipeline import BasePipeline
from .prompts import PROMPT_GEN_REASON_QUESTION_PLUS, PROMPT_GEN_REASON_NO_QUESTION_PLUS

class AfterCoTPipeline(BasePipeline):
    def make_user_messages(self, row: dict) -> str:
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            # <보기>가 있을 때
        if row["question_plus"]:
            user_message = PROMPT_GEN_REASON_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_GEN_REASON_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )
            
        return user_message
