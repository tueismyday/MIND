"""
Vector database management for the MIND system.

This module provides a centralized database manager that handles connections
to three ChromaDB vector databases: patient records, clinical guidelines, and
generated documents. Uses lazy loading for efficient resource management and
singleton pattern for the database manager instance.

Architecture:
    The DatabaseManager class manages three separate ChromaDB instances:
    - Patient DB: Stores patient medical record chunks
    - Guideline DB: Stores clinical guideline chunks
    - Generated Docs DB: Stores previously generated documentation

    Each database is initialized lazily on first access to minimize startup
    time and memory usage. The embedding model is loaded once and shared
    across all databases.

Key Classes:
    DatabaseManager: Singleton manager for all vector database instances

Dependencies:
    - langchain_chroma: Vector database interface
    - core.embeddings: Embedding model for vectorization
    - core.reranker: Cross-encoder for relevance scoring

Thread Safety:
    The current implementation uses module-level singleton (db_manager) which
    is not thread-safe. For multi-threaded applications, consider adding
    locking mechanisms or using separate instances per thread.

Example:
    >>> from core.database import db_manager
    >>> # Access patient database (lazy initialization)
    >>> patient_db = db_manager.patient_db
    >>> results = patient_db.similarity_search("diabetes", k=5)
    >>>
    >>> # Get database statistics
    >>> stats = db_manager.get_database_statistics()
    >>> print(f"Patient records: {stats['patient_record_chunks']}")
"""

import logging
from typing import Optional

from langchain_chroma import Chroma

from .embeddings import get_embedding_model
from .reranker import get_reranker_model
from .exceptions import DatabaseConnectionError
from .types import DatabaseStats
from config.settings import (
    GUIDELINE_DB_DIR,
    PATIENT_DB_DIR,
    GENERATED_DOCS_DB_DIR,
    ensure_directories
)

# Configure logger for this module
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all vector database instances for the MIND system.

    This class provides centralized management of ChromaDB vector databases
    with lazy initialization. All databases share the same embedding model
    for consistency.

    Attributes:
        embeddings: Shared embedding model for all databases
        reranker: Shared cross-encoder reranker model

    Properties:
        patient_db: Patient records vector database (lazy loaded)
        guideline_db: Clinical guidelines vector database (lazy loaded)
        generated_docs_db: Generated documents vector database (lazy loaded)

    Example:
        >>> manager = DatabaseManager()
        >>> patient_db = manager.patient_db
        >>> results = patient_db.similarity_search("condition", k=3)
    """

    def __init__(self):
        """
        Initialize the database manager.

        Ensures database directories exist and loads the shared embedding
        and reranker models. The actual database connections are created
        lazily on first access.

        Raises:
            DatabaseConnectionError: If directory creation fails
            ModelLoadingError: If model loading fails
        """
        logger.info("Initializing DatabaseManager")

        # Ensure all database directories exist
        try:
            ensure_directories()
            logger.debug("Database directories verified/created")
        except Exception as e:
            logger.error(f"Failed to create database directories: {e}")
            raise DatabaseConnectionError(
                f"Failed to create database directories: {e}"
            )

        # Load embedding model at startup (shared across all databases)
        logger.info("Loading shared embedding model")
        self.embeddings = get_embedding_model()
        logger.info("Embedding model loaded successfully")

        # Pre-load reranker model at startup (while GPU memory still available)
        # This prevents OOM errors when reranker loads lazily during first retrieval
        logger.info("Pre-loading reranker model at startup")
        self.reranker = get_reranker_model()
        logger.info("Reranker model loaded successfully")

        # Initialize database references (lazy loading)
        self._patient_db: Optional[Chroma] = None
        self._guideline_db: Optional[Chroma] = None
        self._generated_docs_db: Optional[Chroma] = None

        logger.info("DatabaseManager initialization complete")

    def _get_or_create_db(
        self,
        db_name: str,
        persist_directory: str,
        cached_instance: Optional[Chroma]
    ) -> Chroma:
        """
        Generic method to get or create a database instance.

        This method implements the lazy loading pattern for database instances,
        reducing code duplication across the property methods.

        Args:
            db_name: Name of the database for logging
            persist_directory: Directory path for ChromaDB persistence
            cached_instance: Current cached instance (None if not yet loaded)

        Returns:
            ChromaDB instance

        Raises:
            DatabaseConnectionError: If database initialization fails
        """
        if cached_instance is None:
            logger.debug(f"Initializing {db_name} database from {persist_directory}")
            try:
                cached_instance = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"{db_name} database initialized successfully")
            except Exception as e:
                logger.error(
                    f"Failed to initialize {db_name} database: {e}",
                    exc_info=True
                )
                raise DatabaseConnectionError(
                    f"Failed to initialize database: {e}",
                    db_type=db_name
                )
        return cached_instance

    @property
    def patient_db(self) -> Chroma:
        """
        Get the patient records vector database.

        Lazy loads the database on first access. All patient medical records
        are stored as chunks in this database.

        Returns:
            ChromaDB instance for patient records

        Raises:
            DatabaseConnectionError: If database initialization fails
        """
        if self._patient_db is None:
            self._patient_db = self._get_or_create_db(
                db_name="patient",
                persist_directory=str(PATIENT_DB_DIR),
                cached_instance=self._patient_db
            )
        return self._patient_db

    @property
    def guideline_db(self) -> Chroma:
        """
        Get the clinical guidelines vector database.

        Lazy loads the database on first access. Clinical guidelines and
        medical knowledge are stored as chunks in this database.

        Returns:
            ChromaDB instance for clinical guidelines

        Raises:
            DatabaseConnectionError: If database initialization fails
        """
        if self._guideline_db is None:
            self._guideline_db = self._get_or_create_db(
                db_name="guideline",
                persist_directory=str(GUIDELINE_DB_DIR),
                cached_instance=self._guideline_db
            )
        return self._guideline_db

    @property
    def generated_docs_db(self) -> Chroma:
        """
        Get the generated documents vector database.

        Lazy loads the database on first access. Previously generated
        documentation is stored in this database for reference and reuse.

        Returns:
            ChromaDB instance for generated documents

        Raises:
            DatabaseConnectionError: If database initialization fails
        """
        if self._generated_docs_db is None:
            self._generated_docs_db = self._get_or_create_db(
                db_name="generated_docs",
                persist_directory=str(GENERATED_DOCS_DB_DIR),
                cached_instance=self._generated_docs_db
            )
        return self._generated_docs_db

    def get_database_statistics(self) -> DatabaseStats:
        """
        Get document count statistics for all databases.

        Queries each database to retrieve the number of document chunks stored.
        Returns 0 for databases that fail to respond or don't exist yet.

        Returns:
            Dictionary with database names as keys and document counts as values:
                - patient_record_chunks: Number of patient record chunks
                - guideline_chunks: Number of guideline chunks
                - generated_document_chunks: Number of generated document chunks

        Example:
            >>> stats = db_manager.get_database_statistics()
            >>> print(f"Patient records: {stats['patient_record_chunks']}")
        """
        logger.debug("Retrieving database statistics")

        stats: DatabaseStats = {}

        # Get patient database count
        try:
            stats['patient_record_chunks'] = self.patient_db._collection.count()
            logger.debug(f"Patient DB: {stats['patient_record_chunks']} chunks")
        except Exception as e:
            logger.warning(f"Failed to get patient DB count: {e}")
            stats['patient_record_chunks'] = 0

        # Get guideline database count
        try:
            stats['guideline_chunks'] = self.guideline_db._collection.count()
            logger.debug(f"Guideline DB: {stats['guideline_chunks']} chunks")
        except Exception as e:
            logger.warning(f"Failed to get guideline DB count: {e}")
            stats['guideline_chunks'] = 0

        # Get generated documents database count
        try:
            stats['generated_document_chunks'] = self.generated_docs_db._collection.count()
            logger.debug(f"Generated Docs DB: {stats['generated_document_chunks']} chunks")
        except Exception as e:
            logger.warning(f"Failed to get generated docs DB count: {e}")
            stats['generated_document_chunks'] = 0

        logger.debug(f"Database statistics retrieved: {stats}")
        return stats

    def display_database_statistics(self) -> None:
        """
        Display database statistics to console.

        Retrieves and prints document counts for all three databases in a
        user-friendly format. This is useful for debugging and system status
        monitoring.

        Example:
            >>> db_manager.display_database_statistics()
            [INFO] Database Statistics
            [INFO] ==================
            [INFO] Patient records: 1,234 chunks
            [INFO] Guidelines: 5,678 chunks
            [INFO] Generated documents: 42 chunks
        """
        try:
            stats = self.get_database_statistics()

            logger.info("\n" + "=" * 60)
            logger.info("Database Statistics")
            logger.info("=" * 60)
            logger.info(
                f"Patient records: {stats['patient_record_chunks']:,} chunks"
            )
            logger.info(
                f"Guidelines: {stats['guideline_chunks']:,} chunks"
            )
            logger.info(
                f"Generated documents: {stats['generated_document_chunks']:,} chunks"
            )
            logger.info("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Failed to display database statistics: {e}", exc_info=True)


# Global database manager instance (singleton pattern)
# Note: This is not thread-safe. For multi-threaded applications,
# consider using thread-local storage or separate instances per thread.
db_manager = DatabaseManager()
