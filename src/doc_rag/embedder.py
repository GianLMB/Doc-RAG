import logging

import chromadb
from sentence_transformers import SentenceTransformer

from .utils import setup_logger


class DocumentEmbedder:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        embedder_name: str,
        log_level: int = logging.INFO,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedder_name = embedder_name
        self.logger = setup_logger(self, level=log_level)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding model
        self.logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedder_name)

    def chunk_text(
        self, text: str, chunk_size: int = 300, overlap: int = 50
    ) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def embed_documents(self, documents: list[dict[str, str]]):
        """Embed and store documents in ChromaDB."""
        all_chunks = []
        all_metadatas = []
        all_ids = []

        self.logger.info("Chunking documents...")
        for doc_idx, doc in enumerate(documents):
            self.logger.debug(f"Chunking document {doc_idx} - {doc['title']}")
            chunks = self.chunk_text(doc["content"])

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append(
                    {"url": doc["url"], "title": doc["title"], "chunk_index": chunk_idx}
                )
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")

        print(f"Embedding {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            all_chunks, show_progress_bar=True, convert_to_numpy=True
        )

        self.logger.info(f"Storing in ChromaDB at {self.db_path}")
        # Add in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            self.logger.debug(f"Adding batch {i // batch_size + 1}")
            batch_end = min(i + batch_size, len(all_chunks))
            self.collection.add(
                documents=all_chunks[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                metadatas=all_metadatas[i:batch_end],
                ids=all_ids[i:batch_end],
            )

        self.logger.info(
            f"Successfully embedded {len(all_chunks)} chunks from {len(documents)} documents"
        )

    @staticmethod
    def clear_collection(db_path: str, collection_name: str):
        """Clear all documents from the collection."""
        client = chromadb.PersistentClient(path=db_path)
        if collection_name not in [col.name for col in client.list_collections()]:
            return False
        client.delete_collection(collection_name)
        return True
