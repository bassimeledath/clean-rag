import weave
from models import EmbeddingModel, TextModel
from vector_db import VectorDB
from utils import read_text_file, chunk_text, batched
from tqdm import tqdm
import numpy as np


class RAGQueryEngine:
    def __init__(self):
        self.text_model = TextModel()
        self.embedding_model = EmbeddingModel()
        self.vector_db = VectorDB()
        self.vector_db.load()

    @weave.op()
    def get_most_relevant_document(self, query):
        query_embedding = self.embedding_model.get_embeddings([query])[0]
        results = self.vector_db.search(query_embedding)
        return results[0]['text'] if results else ""

    @weave.op()
    def predict(self, question: str) -> dict:
        context = self.get_most_relevant_document(question)
        prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
        answer = self.text_model.generate_text(prompt)
        return {'answer': answer, 'context': context}


class DocumentProcessor:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_db = VectorDB()

    @weave.op()
    def process_file(self, filename):
        text = read_text_file(filename)
        chunks = list(chunk_text(text, filename))
        return chunks

    @weave.op()
    def embed_and_store(self, chunks):
        for batch in tqdm(batched(chunks), desc="Processing chunks"):
            texts = [chunk.text for chunk in batch]
            embeddings = self.embedding_model.get_embeddings(texts)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = np.array(embedding).astype('float32')
            self.vector_db.insert(batch)

    @weave.op()
    def process_and_embed_documents(self, filename):
        chunks = self.process_file(filename)
        self.embed_and_store(chunks)
        self.vector_db.save()  # Save the index and chunks
        print(f"Processed and embedded documents from {filename}")
