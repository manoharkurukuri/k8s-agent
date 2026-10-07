from pydantic import BaseModel


class Explanation(BaseModel):
    topic: str
    summary: str
    example: str
    difficulty: str