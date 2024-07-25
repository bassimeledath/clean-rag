import weave
from pydantic import BaseModel, Field
import uuid
import numpy as np
from typing import Iterable, Optional
from models import TextChunk
from constants import CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE


@weave.op()
def read_text_file(filename):
    with open(filename, 'r') as file:
        return file.read()


@weave.op()
def chunk_text(text: str, filename: str) -> Iterable[TextChunk]:
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = ' '.join(words[i:i + CHUNK_SIZE])
        yield TextChunk(id=i, text=chunk, filename=filename)


@weave.op()
def batched(iterable, n=BATCH_SIZE):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
