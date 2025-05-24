from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pathlib import Path
import faiss


class RepositoryParser:
    """
    Load a folder of files, slice them into sentence-level chunks, and build a
    FAISS-backed `VectorStoreIndex`.

    Typical use
    -----------
    >>> from llama_index.embeddings.openai import OpenAIEmbedding
    >>> parser = RepositoryParser("repository").split_documents()
    >>> v_index = parser.index_documents("text-embedding-3-small")
    """

    def __init__(self, repository_path: str|Path = "repository", dimension: int = 1536, recursive: bool = True) -> None:
        self._path = Path(repository_path)
        self._documents = SimpleDirectoryReader(self._path, recursive=recursive).load_data()

        self._faiss_index = faiss.IndexFlatL2(dimension)
        self._vector_store = FaissVectorStore(faiss_index=self._faiss_index)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    def split_documents(self, chunk_size: int = 1000, chunk_overlap: int = 350) -> "RepositoryParser":
        """Chunk raw documents into nodes in-place (fluent interface)."""
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._nodes = splitter.get_nodes_from_documents(self._documents)
        return self

    def index_documents(self, model="text-embedding-3-small") -> VectorStoreIndex:
        """Create (or return) the FAISS-backed vector index."""
        embed_model = OpenAIEmbedding(model=model)
        self._index = VectorStoreIndex(self._nodes, storage_context=self._storage_context, embed_model=embed_model)
        return self._index