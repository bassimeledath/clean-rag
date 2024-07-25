import weave
from openai import OpenAI
from constants import SYSTEM_PROMPT
import os
from pydantic import BaseModel, Field
import uuid
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class TextChunk(BaseModel):
    id: int
    text: str
    embedding: Optional[np.ndarray] = None
    filename: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))

    class Config:
        arbitrary_types_allowed = True


class Model:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class TextModel(Model):
    @weave.op()
    def generate_text(self, prompt, model="gpt-3.5-turbo"):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.0
        )
        return completion.choices[0].message.content


class EmbeddingModel(Model):
    @weave.op()
    def get_embeddings(self, texts):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
