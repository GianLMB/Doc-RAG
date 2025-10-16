import logging
from collections.abc import Iterator

import ollama
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from .utils import pull_ollama_model, setup_logger

# COMPLETION_PROMPT = """Based on the following documentation excerpts, answer the question.
# If the answer cannot be found in the documentation, say so.
# Do not add any information that is not in the excerpts. Be concise and to the point.
# If the question asks to implement code snippets, provide the code snippets only without any explanation.

# Documentation:
# {}

# Question: {}

# Answer:"""


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on documentation.
If the answer cannot be found in the documentation, say so.
Do not add any information that is not in the documentation. Be concise and to the point.
If the question asks to implement code snippets, provide the code snippets only without any explanation.

Here is the documentation:
{}
"""


class RAGRetriever:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        model: str,
        embedder_name: str,
        num_results: int = 5,
        log_level: int = logging.INFO,
    ):
        # # Check if model is available
        # available_models = ollama.list().get("models", [])
        # available_models = (
        #     [m["model"] for m in available_models] if available_models else []
        # )
        # if model not in available_models:
        #     raise ValueError(
        #         f"Model '{model}' not found. Available models: {available_models}.\nPlease download"
        #         f" it using the Ollama app, or with `ollama pull {model}` in your terminal."
        #     )

        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model
        self.embedder_name = embedder_name
        self.num_results = num_results
        self.logger = setup_logger(self, level=log_level)

        try:
            pull_ollama_model(model)
        except Exception as e:
            self.logger.warning(f"Error pulling model {model}: {e}")

        # Initialize ChromaDB
        self.logger.info(f"Connecting to ChromaDB in {db_path}")
        self.client = PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedder_name)

        # set context to None
        self.context = None

    def retrieve_context(self, query: str) -> list[dict]:
        """Retrieve relevant document chunks."""

        self.logger.info("Encoding query...")
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        self.logger.info("Retrieving relevant documents...")
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=self.num_results
        )

        self.logger.info("Generating contexts...")
        contexts = []
        for i in range(len(results["documents"][0])):
            contexts.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                    if "distances" in results
                    else None,
                }
            )

        return contexts

    # def generate_response(self, query: str) -> dict[str, any]:
    #     """Generate response using RAG."""
    #     # Retrieve relevant contexts
    #     contexts = self.retrieve_context(query)

    #     # Build prompt
    #     context_text = "\n\n".join(
    #         [
    #             f"[Source: {ctx['metadata']['title']} - {ctx['metadata']['url']}]\n{ctx['content']}"
    #             for ctx in contexts
    #         ]
    #     )

    #     prompt = COMPLETION_PROMPT.format(context_text, query)

    #     # Generate response with Ollama
    #     print(f"Generating response with {self.model}...")
    #     response = ollama.generate(model=self.model, prompt=prompt)

    #     return {
    #         "answer": response["response"],
    #         "sources": contexts,
    #         "model": self.model,
    #     }

    def chat(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Chat with conversation history."""
        if conversation_history is None:
            conversation_history = []

        # Retrieve context
        contexts = self.retrieve_context(query)
        self.context = contexts
        context_text = "\n\n".join([ctx["content"] for ctx in contexts])

        # Build messages
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context_text),
            }
        ]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})

        # Generate response
        stream_response: ollama.ChatResponse = ollama.chat(
            model=self.model, messages=messages, stream=True
        )

        for chunk in stream_response:
            if chunk["message"]:
                yield chunk["message"]["content"]
