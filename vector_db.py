import weave
import numpy as np
import faiss
import pickle
import os


class VectorDB:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.index = None
        self.chunks = []

    def _create_index_if_not_exists(self, dimension):
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

    @weave.op()
    def insert(self, chunks):
        if not chunks:
            return

        # Create index if it doesn't exist
        first_chunk = chunks[0]
        if first_chunk.embedding is not None:
            dimension = len(first_chunk.embedding)
            self._create_index_if_not_exists(dimension)

        # Add vectors to the index
        vectors = [
            chunk.embedding for chunk in chunks if chunk.embedding is not None]
        if vectors:
            self.index.add(np.array(vectors))

        # Store chunks for later retrieval
        self.chunks.extend(chunks)

    @weave.op()
    def search(self, query_embedding, k=5):
        if self.index is None:
            return []

        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        results = [
            {
                "id": self.chunks[i].id,
                "text": self.chunks[i].text,
                "filename": self.chunks[i].filename,
                "uuid": self.chunks[i].uuid,
                "distance": distances[0][j]
            }
            for j, i in enumerate(indices[0])
        ]

        return results

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}.chunks", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load(self):
        if os.path.exists(f"{self.index_path}.index"):
            self.index = faiss.read_index(f"{self.index_path}.index")
        if os.path.exists(f"{self.index_path}.chunks"):
            with open(f"{self.index_path}.chunks", 'rb') as f:
                self.chunks = pickle.load(f)
