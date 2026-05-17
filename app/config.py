# app/config.py
import os
import json
import boto3
import logging
import urllib.parse
from enum import Enum
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.vector_store.factory import get_vector_store

load_dotenv(find_dotenv())


class VectorDBType(Enum):
    PGVECTOR = "pgvector"
    ATLAS_MONGO = "atlas-mongo"


class EmbeddingsProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    HUGGINGFACETEI = "huggingfacetei"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    GOOGLE_GENAI = "google_genai"
    GOOGLE_VERTEXAI = "vertexai"


def get_env_variable(
    var_name: str, default_value: str = None, required: bool = False
) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None and required:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value


RAG_HOST = os.getenv("RAG_HOST", "0.0.0.0")
RAG_PORT = int(os.getenv("RAG_PORT", 8000))

RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "./uploads/")
if not os.path.exists(RAG_UPLOAD_DIR):
    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

VECTOR_DB_TYPE = VectorDBType(
    get_env_variable("VECTOR_DB_TYPE", VectorDBType.PGVECTOR.value)
)
POSTGRES_USE_UNIX_SOCKET = (
    get_env_variable("POSTGRES_USE_UNIX_SOCKET", "False").lower() == "true"
)
POSTGRES_DB = get_env_variable("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_variable("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD", "mypassword")
DB_HOST = get_env_variable("DB_HOST", "db")
DB_PORT = get_env_variable("DB_PORT", "5432")
PGVECTOR_CREATE_EXTENSION = get_env_variable(
    "PGVECTOR_CREATE_EXTENSION", "True"
).lower() in ("true", "1", "yes", "on")
PG_POOL_PRE_PING = get_env_variable("PG_POOL_PRE_PING", "True").lower() in (
    "true",
    "1",
    "yes",
    "on",
)
PG_POOL_RECYCLE = int(get_env_variable("PG_POOL_RECYCLE", "-1"))
POSTGRES_SCHEMA = get_env_variable("POSTGRES_SCHEMA", None) or None
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")
ATLAS_MONGO_DB_URI = get_env_variable(
    "ATLAS_MONGO_DB_URI", "mongodb://127.0.0.1:27018/LibreChat"
)
ATLAS_SEARCH_INDEX = get_env_variable("ATLAS_SEARCH_INDEX", "vector_index")
MONGO_VECTOR_COLLECTION = get_env_variable(
    "MONGO_VECTOR_COLLECTION", None
)  # Deprecated, backwards compatability
CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))

# Markdown-aware chunking.
#
# When enabled, .md files are first split by header boundaries
# (MarkdownHeaderTextSplitter), then any oversized sections are further
# split by RecursiveCharacterTextSplitter as a secondary pass. The full
# header path (e.g. "Modules > Client Presentation") is prepended to each
# chunk's page_content so it ends up in the embedding — a query for "Client
# Presentation module" then maps cleanly onto chunks from that specific
# section rather than competing semantically against unrelated text that
# happens to mention "client" or "presentation."
#
# Defaults to True because the only files this affects are markdown, and
# header-aware chunking is strictly better than character-counted chunking
# for any markdown document with meaningful headers. Set
# MARKDOWN_AWARE_CHUNKING=false to fall back to RecursiveCharacterTextSplitter
# for .md files (e.g. for a deploy that doesn't want the behavior change,
# or to A/B compare retrieval quality).
MARKDOWN_AWARE_CHUNKING = get_env_variable(
    "MARKDOWN_AWARE_CHUNKING", "True"
).lower() in ("true", "1", "yes", "on")

# Header levels to treat as split boundaries. Comma-separated, in order of
# nesting depth. Defaults to "H2,H3" — H1 is typically the document title
# (not a section boundary) and H4+ is usually too fine-grained to justify
# splitting into its own chunk. Override to "H1,H2,H3" if your markdown
# docs use # for sections rather than ##.
def _parse_md_headers(raw: str) -> list[tuple[str, str]]:
    """Parse the env var into the (marker, metadata_key) list that
    MarkdownHeaderTextSplitter expects. e.g. 'H2,H3' →
    [('##', 'Header 2'), ('###', 'Header 3')]. Unknown tokens are dropped
    with a warning rather than crashing the service at boot. The module-
    level `logger` is defined further down in this file (post-app
    construction), so we use the stdlib logger directly here to avoid an
    import-order trap."""
    marker_by_level = {
        "H1": ("#", "Header 1"),
        "H2": ("##", "Header 2"),
        "H3": ("###", "Header 3"),
        "H4": ("####", "Header 4"),
        "H5": ("#####", "Header 5"),
        "H6": ("######", "Header 6"),
    }
    out = []
    for token in (t.strip().upper() for t in raw.split(",")):
        if token in marker_by_level:
            out.append(marker_by_level[token])
        elif token:
            logging.getLogger(__name__).warning(
                "MARKDOWN_HEADERS_TO_SPLIT_ON: ignoring unknown header token %r "
                "(expected one of H1-H6)", token,
            )
    return out


MARKDOWN_HEADERS_TO_SPLIT_ON = _parse_md_headers(
    get_env_variable("MARKDOWN_HEADERS_TO_SPLIT_ON", "H2,H3")
)

# Contextual retrieval — prepend LLM-generated context to each chunk before
# embedding. Disabled by default; flip to true once deployed and verified.
CONTEXTUAL_RETRIEVAL_ENABLED = get_env_variable(
    "CONTEXTUAL_RETRIEVAL_ENABLED", "False"
).lower() in ("true", "1", "yes", "on")
CONTEXTUALIZER_PROVIDER = get_env_variable("CONTEXTUALIZER_PROVIDER", "anthropic")
CONTEXTUALIZER_MODEL = get_env_variable(
    "CONTEXTUALIZER_MODEL", "claude-haiku-4-5-20251001"
)
ANTHROPIC_API_KEY = get_env_variable("ANTHROPIC_API_KEY", "")
CONTEXTUALIZER_API_KEY = get_env_variable("CONTEXTUALIZER_API_KEY", "") or ANTHROPIC_API_KEY
CONTEXTUALIZER_BASE_URL = get_env_variable("CONTEXTUALIZER_BASE_URL", None)
CONTEXTUALIZER_MAX_CONCURRENCY = int(
    get_env_variable("CONTEXTUALIZER_MAX_CONCURRENCY", "5")
)
MAX_CHUNKS_PER_CONTEXTUALIZE = int(
    get_env_variable("MAX_CHUNKS_PER_CONTEXTUALIZE", "200")
)

# Batch processing configuration for memory-constrained environments.
# When EMBEDDING_BATCH_SIZE > 0, documents are processed in batches to reduce
# peak memory usage. This is useful for Kubernetes pods with memory limits.
#
# Trade-offs:
# - Smaller batch size = lower memory, more DB round trips
# - Larger batch size = higher memory, fewer DB round trips
# - 0 = disable batching, process all at once
#
# Default of 500 is conservative and works well for most embedding providers.
# Increase to 750 for higher throughput at the cost of higher peak memory.
EMBEDDING_BATCH_SIZE = int(get_env_variable("EMBEDDING_BATCH_SIZE", "500"))

# Maximum number of batches to buffer in memory during async processing.
# Higher values allow more parallelism but use more memory.
EMBEDDING_MAX_QUEUE_SIZE = int(get_env_variable("EMBEDDING_MAX_QUEUE_SIZE", "3"))

env_value = get_env_variable("PDF_EXTRACT_IMAGES", "False").lower()
PDF_EXTRACT_IMAGES = True if env_value == "true" else False

if POSTGRES_USE_UNIX_SOCKET:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@/{urllib.parse.quote_plus(POSTGRES_DB)}?host={urllib.parse.quote_plus(DB_HOST)}"
else:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@{DB_HOST}:{DB_PORT}/{urllib.parse.quote_plus(POSTGRES_DB)}"

CONNECTION_STRING = f"postgresql+psycopg2://{connection_suffix}"
DSN = f"postgresql://{connection_suffix}"

## Logging

HTTP_RES = "http_res"
HTTP_REQ = "http_req"

logger = logging.getLogger()

debug_mode = os.getenv("DEBUG_RAG_API", "False").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)
console_json = get_env_variable("CONSOLE_JSON", "False").lower() == "true"

if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if console_json:

    class JsonFormatter(logging.Formatter):
        def __init__(self):
            super(JsonFormatter, self).__init__()

        def format(self, record):
            json_record = {}

            json_record["message"] = record.getMessage()

            if HTTP_REQ in record.__dict__:
                json_record[HTTP_REQ] = record.__dict__[HTTP_REQ]

            if HTTP_RES in record.__dict__:
                json_record[HTTP_RES] = record.__dict__[HTTP_RES]

            if record.levelno == logging.ERROR and record.exc_info:
                json_record["exception"] = self.formatException(record.exc_info)

            timestamp = datetime.fromtimestamp(record.created)
            json_record["timestamp"] = timestamp.isoformat()

            # add level
            json_record["level"] = record.levelname
            json_record["filename"] = record.filename
            json_record["lineno"] = record.lineno
            json_record["funcName"] = record.funcName
            json_record["module"] = record.module
            json_record["threadName"] = record.threadName

            return json.dumps(json_record)

    formatter = JsonFormatter()
else:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

handler = logging.StreamHandler()  # or logging.FileHandler("app.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        logger_method = logger.info

        if str(request.url).endswith("/health"):
            logger_method = logger.debug

        logger_method(
            f"Request {request.method} {request.url} - {response.status_code}",
            extra={
                HTTP_REQ: {"method": request.method, "url": str(request.url)},
                HTTP_RES: {"status_code": response.status_code},
            },
        )

        return response


logging.getLogger("uvicorn.access").disabled = True

## Credentials

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY", "")
RAG_OPENAI_API_KEY = get_env_variable("RAG_OPENAI_API_KEY", OPENAI_API_KEY)
RAG_OPENAI_BASEURL = get_env_variable("RAG_OPENAI_BASEURL", None)
RAG_OPENAI_PROXY = get_env_variable("RAG_OPENAI_PROXY", None)
AZURE_OPENAI_API_KEY = get_env_variable("AZURE_OPENAI_API_KEY", "")
RAG_AZURE_OPENAI_API_VERSION = get_env_variable("RAG_AZURE_OPENAI_API_VERSION", None)
RAG_AZURE_OPENAI_API_KEY = get_env_variable(
    "RAG_AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY
)
AZURE_OPENAI_ENDPOINT = get_env_variable("AZURE_OPENAI_ENDPOINT", "")
RAG_AZURE_OPENAI_ENDPOINT = get_env_variable(
    "RAG_AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT
).rstrip("/")
HF_TOKEN = get_env_variable("HF_TOKEN", "")
OLLAMA_BASE_URL = get_env_variable("OLLAMA_BASE_URL", "http://ollama:11434")
AWS_ACCESS_KEY_ID = get_env_variable("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY", "")
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY", "")
GOOGLE_KEY = get_env_variable("GOOGLE_KEY", GOOGLE_API_KEY)
RAG_GOOGLE_API_KEY = get_env_variable("RAG_GOOGLE_API_KEY", GOOGLE_KEY)
AWS_SESSION_TOKEN = get_env_variable("AWS_SESSION_TOKEN", "")
GOOGLE_APPLICATION_CREDENTIALS = get_env_variable("GOOGLE_APPLICATION_CREDENTIALS", "")
env_value = get_env_variable("RAG_CHECK_EMBEDDING_CTX_LENGTH", "True").lower()
RAG_CHECK_EMBEDDING_CTX_LENGTH = True if env_value == "true" else False

# Only parse RAG_DISTANCE_THRESHOLD when it will actually be applied (pgvector).
# Under atlas-mongo the setting is documented as ignored, so parsing it
# unconditionally would turn a stale/non-numeric value into a hard boot failure
# for an unrelated backend.
RAG_DISTANCE_THRESHOLD = None
if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
    _distance_threshold_raw = get_env_variable("RAG_DISTANCE_THRESHOLD", None)
    if _distance_threshold_raw not in (None, ""):
        RAG_DISTANCE_THRESHOLD = float(_distance_threshold_raw)

## Embeddings


def init_embeddings(provider, model, dimensions=None):
    if provider == EmbeddingsProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        kwargs = dict(
            model=model,
            api_key=RAG_OPENAI_API_KEY,
            openai_api_base=RAG_OPENAI_BASEURL,
            openai_proxy=RAG_OPENAI_PROXY,
            chunk_size=EMBEDDINGS_CHUNK_SIZE,
            check_embedding_ctx_length=RAG_CHECK_EMBEDDING_CTX_LENGTH,
        )
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        return OpenAIEmbeddings(**kwargs)
    elif provider == EmbeddingsProvider.AZURE:
        from langchain_openai import AzureOpenAIEmbeddings

        kwargs = dict(
            azure_deployment=model,
            api_key=RAG_AZURE_OPENAI_API_KEY,
            azure_endpoint=RAG_AZURE_OPENAI_ENDPOINT,
            api_version=RAG_AZURE_OPENAI_API_VERSION,
            chunk_size=EMBEDDINGS_CHUNK_SIZE,
            check_embedding_ctx_length=RAG_CHECK_EMBEDDING_CTX_LENGTH,
        )
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        return AzureOpenAIEmbeddings(**kwargs)
    elif provider == EmbeddingsProvider.HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model, encode_kwargs={"normalize_embeddings": True}
        )
    elif provider == EmbeddingsProvider.HUGGINGFACETEI:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(model=model)
    elif provider == EmbeddingsProvider.OLLAMA:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
    elif provider == EmbeddingsProvider.GOOGLE_GENAI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=RAG_GOOGLE_API_KEY or None,
        )
    elif provider == EmbeddingsProvider.GOOGLE_VERTEXAI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=RAG_GOOGLE_API_KEY or None,
            vertexai=True,
            project=get_env_variable("GOOGLE_CLOUD_PROJECT", None),
            location=get_env_variable("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    elif provider == EmbeddingsProvider.BEDROCK:
        from langchain_aws import BedrockEmbeddings

        session_kwargs = {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "region_name": AWS_DEFAULT_REGION,
        }

        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

        session = boto3.Session(**session_kwargs)
        return BedrockEmbeddings(
            client=session.client("bedrock-runtime"),
            model_id=model,
            region_name=AWS_DEFAULT_REGION,
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")


EMBEDDINGS_PROVIDER = EmbeddingsProvider(
    get_env_variable("EMBEDDINGS_PROVIDER", EmbeddingsProvider.OPENAI.value).lower()
)

# Only parse EMBEDDINGS_DIMENSIONS for providers that honor it (OpenAI / Azure).
# Parsing unconditionally at import would turn an unrelated stale env var
# (e.g. EMBEDDINGS_DIMENSIONS=foo left over from an OpenAI deployment) into a
# hard boot failure under bedrock / hf / ollama / etc., even though those
# providers silently ignore the value.
EMBEDDINGS_DIMENSIONS = None

if EMBEDDINGS_PROVIDER in (EmbeddingsProvider.OPENAI, EmbeddingsProvider.AZURE):
    _embeddings_dimensions_raw = get_env_variable("EMBEDDINGS_DIMENSIONS", None)
    if _embeddings_dimensions_raw not in (None, ""):
        EMBEDDINGS_DIMENSIONS = int(_embeddings_dimensions_raw)

if EMBEDDINGS_PROVIDER == EmbeddingsProvider.OPENAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
    # 1000 is the default chunk size for OpenAI, but this causes API rate limits to be hit
    EMBEDDINGS_CHUNK_SIZE = get_env_variable("EMBEDDINGS_CHUNK_SIZE", 200)
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.AZURE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
    # 2048 is the default (and maximum) chunk size for Azure, but this often causes unexpected 429 errors
    EMBEDDINGS_CHUNK_SIZE = get_env_variable("EMBEDDINGS_CHUNK_SIZE", 200)
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACE:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACETEI:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "http://huggingfacetei:3000"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.GOOGLE_VERTEXAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "gemini-embedding-001")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.OLLAMA:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "nomic-embed-text")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.GOOGLE_GENAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "gemini-embedding-001")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.BEDROCK:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "amazon.titan-embed-text-v1"
    )
    AWS_DEFAULT_REGION = get_env_variable("AWS_DEFAULT_REGION", "us-east-1")
else:
    raise ValueError(f"Unsupported embeddings provider: {EMBEDDINGS_PROVIDER}")

embeddings = init_embeddings(
    EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL, dimensions=EMBEDDINGS_DIMENSIONS
)

logger.info(f"Initialized embeddings of type: {type(embeddings)}")

# Vector store
if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
    vector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="async",
        create_extension=PGVECTOR_CREATE_EXTENSION,
        pool_pre_ping=PG_POOL_PRE_PING,
        pool_recycle=PG_POOL_RECYCLE,
        schema=POSTGRES_SCHEMA,
    )
elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
    # Backward compatability check
    if MONGO_VECTOR_COLLECTION:
        logger.info(
            f"DEPRECATED: Please remove env var MONGO_VECTOR_COLLECTION and instead use COLLECTION_NAME and ATLAS_SEARCH_INDEX. You can set both as same, but not neccessary. See README for more information."
        )
        ATLAS_SEARCH_INDEX = MONGO_VECTOR_COLLECTION
        COLLECTION_NAME = MONGO_VECTOR_COLLECTION
    vector_store = get_vector_store(
        connection_string=ATLAS_MONGO_DB_URI,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="atlas-mongo",
        search_index=ATLAS_SEARCH_INDEX,
    )
else:
    raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")

retriever = vector_store.as_retriever()

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
    "yml",
    "yaml",
    "eml",
    "ex",
    "exs",
    "erl",
    "tsx",
    "jsx",
    "lhs",
]
