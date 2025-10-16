import logging
import os
from functools import wraps

import chromadb
import ollama

_DEFAULT_DB_PATH = "~/.cache/doc-rag-db"
_DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"
_DEFAULT_OLLAMA_MODEL = "gemma3:4b"
_DEFAULT_LOG_LEVEL = logging.INFO


def get_defaults():
    return {
        "db_path": os.path.expanduser(os.getenv("RAG_DOC_DB_PATH", _DEFAULT_DB_PATH)),
        "embedder_name": os.getenv("EMBEDDER_NAME", _DEFAULT_EMBEDDER),
        "ollama_model": os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL),
        "log_level": int(os.getenv("LOG_LEVEL", _DEFAULT_LOG_LEVEL)),
    }


def pull_ollama_model(model: str):
    """Pull Ollama model if not already available."""
    available_models = ollama.list().get("models", [])
    available_models = (
        [m["model"] for m in available_models] if available_models else []
    )
    if model not in available_models:
        print(
            f"Model '{model}' not found. Available models: {available_models}.\nPulling the model..."
        )
        ollama.pull(model)
        print(f"✓ Model '{model}' pulled successfully!")


def list_chromadb_collections(db_path: str) -> list[str]:
    """List collections in ChromaDB at the given path."""
    client = chromadb.PersistentClient(path=db_path)
    return [col.name for col in client.list_collections()]


def setup_logger(object: object, level: int = logging.INFO):
    logger = logging.getLogger(object.__class__.__name__)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def stream_to_gradio(level: int = logging.INFO, logger_names: list[str] | None = None):
    """Decorator to stream yields and specific loggers to Gradio.

    Args:
        level: Logging level to capture
        logger_names: List of logger names to capture, or None for root logger
                     Examples: ['DocumentationScraper', 'DocumentEmbedder']
                              ['myapp.scraper', 'myapp.embedder']
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_messages = []

            class GradioLogHandler(logging.Handler):
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        log_messages.append(msg)
                    except Exception:
                        self.handleError(record)

            # Setup logging handler
            handler = GradioLogHandler()
            formatter = logging.Formatter("%(name)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(level)

            # Store original logger levels
            original_levels = {}
            loggers_to_restore = []

            # Add handler to specific loggers or root
            if logger_names:
                for logger_name in logger_names:
                    logger = logging.getLogger(logger_name)
                    original_levels[logger_name] = logger.level
                    logger.setLevel(level)
                    logger.addHandler(handler)
                    loggers_to_restore.append(logger)
            else:
                # Fallback to root logger
                root_logger = logging.getLogger()
                original_levels["root"] = root_logger.level
                root_logger.setLevel(level)
                root_logger.addHandler(handler)
                loggers_to_restore.append(root_logger)

            try:
                gen = func(*args, **kwargs)

                for yielded_value in gen:
                    output = ""
                    if log_messages:
                        output = "\n".join(log_messages) + "\n\n"
                    if yielded_value:
                        output += yielded_value
                    yield output

                if log_messages:
                    yield "\n".join(log_messages)

            except Exception as e:
                output = ""
                if log_messages:
                    output = "\n".join(log_messages) + "\n\n"
                output += f"❌ Error: {e!s}"
                yield output

            finally:
                # Restore loggers
                for logger in loggers_to_restore:
                    logger.removeHandler(handler)

                # Restore original levels
                if logger_names:
                    for logger_name in logger_names:
                        logger = logging.getLogger(logger_name)
                        logger.setLevel(original_levels[logger_name])
                else:
                    root_logger = logging.getLogger()
                    root_logger.setLevel(original_levels["root"])

        return wrapper

    return decorator
