from pydantic import BaseModel
from typing import List

class Article(BaseModel):
    title: str
    content: str

class SearchRequest(BaseModel):
    query: str

class ArticleResponse(BaseModel):
    title: str
    content: str
    score: float
