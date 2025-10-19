# Doc-RAG

A basic RAG-based tool to explore software documentation and PDFs in chat mode.

Given an input URL, the program parses the targeted web page and all subpages (including PDF files), then it retrieves, chunks and embeds the contents and constructs a collection in a ChromaDB database. The database can then be queried in chat mode running Ollama models, to ask about specific pieces of information.

The entry points are a command line interface (accessible with the base command `doc-rag`) and a Gradio app, that can be launched with `doc-rag-ui`.

### TODO ✏️

Add options to select subpages to exclude