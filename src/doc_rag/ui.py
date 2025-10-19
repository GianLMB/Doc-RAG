import os

import gradio as gr
from dotenv import load_dotenv

from .embedder import DocumentEmbedder
from .retriever import RAGRetriever
from .scraper import DocumentationScraper

load_dotenv(override=True)
from .utils import get_defaults, list_chromadb_collections, stream_to_gradio

default_inputs = get_defaults()


class DocRAGUI:
    def __init__(self):
        self.default_db_path = default_inputs["db_path"]
        self.default_embedder_name = default_inputs["embedder_name"]
        self.default_model = default_inputs["ollama_model"]
        self.log_level = default_inputs["log_level"]
        self.retriever = None

    @stream_to_gradio(
        level=default_inputs["log_level"],
        logger_names=["DocumentationScraper", "DocumentEmbedder"],
    )
    def index_documentation(self, url, max_pages, db_name, collection, embedder_name):
        """Index documentation from URL."""
        try:
            yield "🔄 Starting scraping process...\n"

            scraper = DocumentationScraper(url, max_pages=int(max_pages))
            documents = scraper.scrape()

            yield f"✓ Scraped {len(documents)} documents\n\n🔄 Embedding documents...\n"

            embedder = DocumentEmbedder(
                db_path=db_name, collection_name=collection, embedder_name=embedder_name
            )
            embedder.embed_documents(documents)

            yield f"✓ Successfully indexed {len(documents)} documents!\n\nYou can now query in the 'Query' tab."

        except Exception as e:
            yield f"❌ Error: {e!s}"

    def list_collections(self, db_path):
        collections = list_chromadb_collections(db_path)
        message = "Available Collections:\n"
        for col in collections:
            message += f"- {col}\n"
        yield message

    def query_documentation(
        self,
        question,
        model_name,
        db_path,
        collection_name,
        embedder_name,
        n_results,
        history,
    ):
        """Query the documentation."""
        try:
            if self.retriever is None or self.retriever.model != model_name:
                # Show loading message
                history.append((question, "🔄 Loading model..."))
                yield history, history

                self.retriever = RAGRetriever(
                    db_path=db_path,
                    collection_name=collection_name,
                    model=model_name,
                    embedder_name=embedder_name,
                )

            # else:
            # Update to show retrieval in progress
            history[-1] = (question, "🔍 Retrieving context...")
            yield history, history

            # else:
            #     history.append((question, ""))

            full_answer = ""

            # stream the response
            for chunk in self.retriever.chat(
                question,
                num_results=int(n_results),  # , conversation_history=history
            ):
                full_answer += chunk
                history[-1] = (question, full_answer)  # update last message
                yield history, history

            # append sources at the end
            response = full_answer + "\n\n**Sources:**\n"
            for i, source in enumerate(self.retriever.context[:3], 1):
                response += f"{i}. [{source[0]}]({source[1]})\n"

            history[-1] = (question, response)
            yield history, history

        except Exception as e:
            error_msg = f"❌ Error: {e!s}"
            history.append((question, error_msg))
            yield history, history

    def launch(self):
        """Launch Gradio interface."""
        with gr.Blocks(title="Documentation RAG") as demo:
            gr.Markdown("# 📚 Documentation RAG Pipeline")
            gr.Markdown("Index and query software documentation using local LLMs")

            with gr.Tab("Index Documentation"):
                gr.Markdown("### Scrape and index documentation")
                with gr.Row():
                    with gr.Column():
                        url_input = gr.Textbox(
                            label="Documentation URL",
                            placeholder="https://docs.example.com",
                            # value="https://docs.python.org/3/library/os.html",
                        )
                        max_pages_input = gr.Slider(
                            minimum=10,
                            maximum=2000,
                            value=100,
                            step=10,
                            label="Maximum Pages",
                        )
                        db_name = gr.Textbox(
                            label="Database Path", value=self.default_db_path
                        )
                        collection = gr.Textbox(
                            label="Collection Name", placeholder="docs"
                        )
                        embedder_name = gr.Textbox(
                            label="Embedder Model", value=self.default_embedder_name
                        )
                        with gr.Row():
                            index_btn = gr.Button(
                                "🚀 Start Indexing", variant="primary"
                            )
                            list_btn = gr.Button("📋 List Collections")

                    with gr.Column():
                        index_output = gr.Textbox(
                            label="Status", lines=20, max_lines=20
                        )

                index_btn.click(
                    fn=self.index_documentation,
                    inputs=[
                        url_input,
                        max_pages_input,
                        db_name,
                        collection,
                        embedder_name,
                    ],
                    outputs=index_output,
                )

                list_btn.click(
                    fn=self.list_collections,
                    inputs=[db_name],
                    outputs=index_output,
                )

            with gr.Tab("Query"):
                gr.Markdown("### Ask questions about the documentation")

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=400)
                        with gr.Row():
                            query_input = gr.Textbox(
                                label="Question",
                                placeholder="Ask a question about the documentation...",
                                scale=4,
                            )
                            submit_btn = gr.Button("Send", scale=1, variant="primary")

                    with gr.Column(scale=1):
                        model_query = gr.Textbox(
                            label="Ollama Model", value=self.default_model
                        )
                        db_path = gr.Textbox(
                            label="Database Path", value=self.default_db_path
                        )
                        collection_name = gr.Textbox(
                            label="Collection Name", placeholder="docs"
                        )
                        embedder_name = gr.Textbox(
                            label="Embedder Model", value=self.default_embedder_name
                        )
                        n_results = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Number of Context Chunks",
                        )
                        clear_btn = gr.Button("Clear Chat")

                state = gr.State([])

                submit_btn.click(
                    fn=self.query_documentation,
                    inputs=[
                        query_input,
                        model_query,
                        db_path,
                        collection_name,
                        embedder_name,
                        n_results,
                        state,
                    ],
                    outputs=[chatbot, state],
                ).then(lambda: "", outputs=query_input)

                query_input.submit(
                    fn=self.query_documentation,
                    inputs=[
                        query_input,
                        model_query,
                        db_path,
                        collection_name,
                        embedder_name,
                        n_results,
                        state,
                    ],
                    outputs=[chatbot, state],
                ).then(lambda: "", outputs=query_input)

                clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

        demo.launch(
            server_name="127.0.0.1",
            server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
            inbrowser=True,
            share=False,
        )


def main():
    """Entry point for UI."""
    ui = DocRAGUI()
    ui.launch()


if __name__ == "__main__":
    main()
