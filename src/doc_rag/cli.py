import click
from dotenv import load_dotenv

from .embedder import DocumentEmbedder
from .retriever import RAGRetriever
from .scraper import DocumentationScraper

load_dotenv(override=True)
from .utils import get_defaults, list_chromadb_collections

default_inputs = get_defaults()


@click.group()
def cli():
    """RAG pipeline for software documentation."""
    pass


@cli.command()
@click.argument("url")
@click.argument("collection")
@click.option(
    "--db-path",
    default=default_inputs["db_path"],
    help="Database path",
)
@click.option("--max-pages", default=100, help="Maximum pages to scrape")
@click.option(
    "--embedder-name",
    default=default_inputs["embedder_name"],
    help="Sentence transformer model name",
)
def index(url, db_path, max_pages, collection, embedder_name):
    """Scrape and index documentation from URL."""
    click.echo(f"Scraping documentation from: {url}")

    # Scrape
    scraper = DocumentationScraper(url, max_pages=max_pages)
    documents = scraper.scrape()

    click.echo(f"\nScraped {len(documents)} documents")

    # Embed and store
    embedder = DocumentEmbedder(
        db_path=db_path, collection_name=collection, embedder_name=embedder_name
    )
    embedder.embed_documents(documents)

    click.echo("\n✓ Indexing complete!")


@cli.command()
@click.argument("collection")
@click.option("--db-path", default=default_inputs["db_path"], help="Database path")
@click.option(
    "--model",
    default=default_inputs["ollama_model"],
    help="Ollama model name",
)
@click.option(
    "--embedder", default=default_inputs["embedder_name"], help="Embedder name"
)
@click.option("--num-results", default=3, help="Number of results to retrieve")
def query(db_path, collection, model, embedder, num_results):
    """Interactive query interface."""
    retriever = RAGRetriever(
        db_path=db_path,
        collection_name=collection,
        model=model,
        embedder_name=embedder,
    )

    click.echo(f"RAG Query Interface (using {model})")
    click.echo("Type 'exit' to quit\n")

    while True:
        question = click.prompt("Question", type=str)

        if question.lower() in ["exit", "quit", "q"]:
            break

        full_answer = ""
        stream_result = retriever.chat(question, num_results=num_results)
        for chunk in stream_result:
            print(chunk, end="", flush=True)
            full_answer += chunk

        click.echo("\n\nSources:")
        for i, source in enumerate(retriever.context[:3], 1):
            click.echo(f"  {i}. {source[0]} - {source[1]}")
        click.echo()


@cli.command()
@click.argument("collection")
@click.option("--db-path", default=default_inputs["db_path"], help="Database path")
def clear(db_path, collection):
    """Clear the document collection."""
    if click.confirm(f"Are you sure you want to clear the collection '{collection}'?"):
        is_removed = DocumentEmbedder.clear_collection(db_path, collection)
        if not is_removed:
            click.echo(f"Collection '{collection}' does not exist.")
        else:
            click.echo("✓ Collection cleared")


@cli.command()
@click.option("--db-path", default=default_inputs["db_path"], help="Database path")
def list_collections(db_path):
    """List collections in the ChromaDB database."""
    collections = list_chromadb_collections(db_path)
    click.echo("Available collections:")
    for col in collections:
        click.echo(f" - {col}")


if __name__ == "__main__":
    cli()
