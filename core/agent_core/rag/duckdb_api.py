import os
import duckdb
import hashlib
import logging
import asyncio
from typing import Dict, Any, List, Optional

from .embedding_utils import get_embedding_provider, EmbeddingProvider

logger = logging.getLogger(__name__)

_db_locks: Dict[str, asyncio.Lock] = {}

class DuckDBRAGStore:
    """
     The Common_Ground_Agent_Core project manages a RAG data source.
    - Asynchronous operations
    - Configuration-driven
    - Supports separation of metadata and embedding data into two tables
    - Supports embedding tables with multiple vector columns
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the RAG store using a configuration dictionary.
        """
        self.config = config
        self.db_file = config.get('database_file')
        if not self.db_file:
            raise ValueError("The configuration dictionary must contain the 'database_file' key.")

        # Create a lock for this database file (if it doesn't already exist)
        if self.db_file not in _db_locks:
            _db_locks[self.db_file] = asyncio.Lock()
        self.db_lock = _db_locks[self.db_file]

        # Get the model provider associated with this configuration
        emb_cfg = self.config.get('embedding_table', {})
        self.model_provider: EmbeddingProvider = get_embedding_provider(
            model_id=emb_cfg.get("emb_model_id"),
            model_config=emb_cfg.get("model_config_params")
        )
        logger.info("duckdb_rag_store_initialized", extra={"database_name": os.path.basename(self.db_file), "embedding_model_id": emb_cfg.get('emb_model_id')})
        
        # Track initialization state - lazy initialization to avoid unawaited coroutine warnings
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """Ensure the database is initialized before any operations. Thread-safe and idempotent."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:  # Double-check after acquiring lock
                return
            await self._initialize_or_check_database()
            self._initialized = True

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Synchronously gets a database connection."""
        try:
            # For both writing and reading, we use non-read-only mode because DuckDB's write operations are very fast
            return duckdb.connect(database=self.db_file, read_only=False)
        except duckdb.Error as e:
            logger.error("duckdb_connection_failed", extra={"database_file": self.db_file, "error": str(e)})
            raise

    async def _execute_in_thread(self, func, *args, **kwargs):
        """Executes synchronous DuckDB operations in a separate thread to avoid blocking the asyncio event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _initialize_or_check_database(self):
        """
        Based on the configured 'database_writable' flag,
        decides whether to initialize (create) the database tables or just check if the table structure exists for a read-only database.
        """
        is_writable = self.config.get('database_writable', False)
        
        if is_writable:
            # --- Behavior 1: For writable databases, ensure tables exist ---
            def _sync_init():
                with self._get_connection() as con:
                    meta_cfg = self.config['meta_table']
                    emb_cfg = self.config['embedding_table']

                    # For internal writable stores, we enforce embedding_column to be a string (single column)
                    emb_col_config = emb_cfg['embedding_column']
                    if not isinstance(emb_col_config, str):
                        raise ValueError(f"For writable data source '{self.config.get('source_name')}', 'embedding_column' must be a single string, not a list.")

                    # Create metadata table
                    con.execute(f"CREATE SEQUENCE IF NOT EXISTS seq_{meta_cfg['name']} START 1;")
                    con.execute(f"""
                        CREATE TABLE IF NOT EXISTS {meta_cfg['name']} (
                            {meta_cfg['id_column']} BIGINT PRIMARY KEY DEFAULT nextval('seq_{meta_cfg['name']}'),
                            project_id VARCHAR, doc_id VARCHAR, url VARCHAR, hash CHAR(64) UNIQUE,
                            meta TEXT, tags VARCHAR[], chunk_text TEXT NOT NULL
                        );
                    """)

                    # Create embedding table
                    con.execute(f"""
                        CREATE TABLE IF NOT EXISTS {emb_cfg['name']} (
                            {emb_cfg['id_column']} BIGINT PRIMARY KEY,
                            "{emb_col_config}" FLOAT[],
                            FOREIGN KEY ({emb_cfg['id_column']}) REFERENCES {meta_cfg['name']}({meta_cfg['id_column']})
                        );
                    """)
                    logger.info("writable_database_schema_verified", extra={"source_name": self.config.get('source_name')})

            await self._execute_in_thread(_sync_init)
        else:
            # --- Behavior 2: For read-only databases, perform a health check ---
            def _sync_check():
                with self._get_connection() as con:
                    meta_cfg = self.config['meta_table']
                    emb_cfg = self.config['embedding_table']
                    
                    # Check if tables exist
                    tables = con.execute("SHOW TABLES;").fetchall()
                    table_names = {t[0] for t in tables}
                    if meta_cfg['name'] not in table_names:
                        raise ConnectionError(f"Health check failed: Metadata table '{meta_cfg['name']}' not found in read-only database '{self.db_file}'.")
                    if emb_cfg['name'] not in table_names:
                        raise ConnectionError(f"Health check failed: Embedding table '{emb_cfg['name']}' not found in read-only database '{self.db_file}'.")

                    # Check if key columns exist
                    meta_cols_result = con.execute(f"DESCRIBE {meta_cfg['name']};").fetchall()
                    meta_cols = {c[0] for c in meta_cols_result}
                    if meta_cfg['id_column'] not in meta_cols:
                         raise ConnectionError(f"Health check failed: ID column '{meta_cfg['id_column']}' not found in table '{meta_cfg['name']}'.")

                    emb_cols_result = con.execute(f"DESCRIBE {emb_cfg['name']};").fetchall()
                    emb_cols = {c[0] for c in emb_cols_result}
                    emb_col_config = emb_cfg['embedding_column']
                    required_emb_cols = emb_col_config if isinstance(emb_col_config, list) else [emb_col_config]
                    for col in required_emb_cols:
                        if col not in emb_cols:
                            raise ConnectionError(f"Health check failed: Embedding column '{col}' not found in table '{emb_cfg['name']}'.")

                    logger.info("readonly_database_health_check_passed", extra={"source_name": self.config.get('source_name')})

            try:
                await self._execute_in_thread(_sync_check)
            except Exception as e:
                logger.error("readonly_database_health_check_failed", extra={"source_name": self.config.get('source_name'), "error": str(e)})
                # You can decide here whether to fail the application due to configuration errors

    async def add_text_chunk(self, chunk_text: str, project_id: str, doc_id: str = None, url: str = None, meta: str = None, tags: list = None) -> Optional[int]:
        """Asynchronously adds a new text chunk to the metadata table."""
        await self._ensure_initialized()
        if not chunk_text:
            raise ValueError("chunk_text cannot be empty.")
        if not self.config.get('database_writable', False):
            raise PermissionError(f"Data source '{self.config.get('source_name')}' is read-only, 'add_text_chunk' operation is not allowed.")
        
        sha256_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        
        async with self.db_lock: # Acquire lock for write operation
            def _sync_add():
                with self._get_connection() as con:
                    meta_table = self.config["meta_table"]["name"]
                    id_col = self.config["meta_table"]["id_column"]
                    
                    existing = con.execute(f"SELECT {id_col} FROM {meta_table} WHERE hash = ? AND project_id = ?", (sha256_hash, project_id)).fetchone()
                    if existing:
                        return existing[0]
                    
                    result = con.execute(f"""
                        INSERT INTO {meta_table} (project_id, doc_id, url, hash, meta, tags, chunk_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING {id_col}
                    """, (project_id, doc_id, url, sha256_hash, meta, tags, chunk_text)).fetchone()
                    return result[0]
            
            new_id = await self._execute_in_thread(_sync_add)
            logger.info("text_chunk_added_or_exists", extra={"chunk_id": new_id})
            return new_id

    async def process_pending_embeddings(self, batch_size: int = 50):
        """Asynchronously generates and stores embeddings for pending text chunks."""
        await self._ensure_initialized()
        if not self.config.get('database_writable', False):
            raise PermissionError(f"Data source '{self.config.get('source_name')}' is read-only, 'process_pending_embeddings' operation is not allowed.")
        
        emb_cfg = self.config['embedding_table']
        embedding_column_config = emb_cfg['embedding_column']
        target_vector_column = embedding_column_config[0] if isinstance(embedding_column_config, list) else embedding_column_config

        while True:
            def _sync_fetch_pending():
                with self._get_connection() as con:
                    meta_cfg = self.config['meta_table']
                    return con.execute(f"""
                        SELECT m.{meta_cfg['id_column']}, m.chunk_text
                        FROM {meta_cfg['name']} m
                        LEFT JOIN {emb_cfg['name']} e ON m.{meta_cfg['id_column']} = e.{emb_cfg['id_column']}
                        WHERE e.{emb_cfg['id_column']} IS NULL
                        LIMIT ?
                    """, (batch_size,)).fetchall()

            pending_rows = await self._execute_in_thread(_sync_fetch_pending)
            if not pending_rows:
                logger.info("No more text chunks to generate embeddings for.")
                break

            ids_in_batch, texts_in_batch = zip(*pending_rows)
            
            task_type = emb_cfg.get('passage_task_type', '')
            embeddings = await self._execute_in_thread(self.model_provider.generate_embedding, list(texts_in_batch), task_type=task_type)

            if embeddings is None or embeddings.size == 0:
                logger.warning("The model provider did not return any embeddings.")
                continue

            async with self.db_lock:
                def _sync_insert_embeddings():
                    with self._get_connection() as con:
                        for row_id, embedding in zip(ids_in_batch, embeddings):
                             con.execute(f"""
                                INSERT INTO {emb_cfg['name']} ({emb_cfg['id_column']}, "{target_vector_column}")
                                VALUES (?, ?)
                                ON CONFLICT ({emb_cfg['id_column']}) DO UPDATE SET
                                "{target_vector_column}" = excluded."{target_vector_column}"
                            """, (row_id, embedding.tolist()))
                        return len(ids_in_batch)
                
                processed_count = await self._execute_in_thread(_sync_insert_embeddings)
                logger.info("embeddings_generated_and_stored", extra={"processed_count": processed_count})

    async def vector_search_text(self, query_text: str, project_id: str, top_k: int = 5, tags: Optional[List[str]] = None) -> List[Dict]:
        """Asynchronously performs a vector search across multiple embedding columns (available for all sources)."""
        await self._ensure_initialized()
        if not query_text or not project_id: return []

        is_global_source = self.config.get('is_global', False)
        if not is_global_source and not project_id:
            logger.warning("non_global_source_missing_project_id", extra={"source_name": self.config.get('source_name')})
            return []
        
        # Asynchronously generate query embedding
        emb_cfg = self.config['embedding_table']
        query_task_type = emb_cfg.get('query_task_type', '')
        query_embedding_array = await self._execute_in_thread(self.model_provider.generate_embedding, [query_text], query_task_type)
        if query_embedding_array is None or query_embedding_array.size == 0:
            return []
        query_embedding = query_embedding_array[0].tolist()
        
        def _sync_search():
            meta_cfg = self.config['meta_table']
            retrieval_cols = meta_cfg.get('retrieval_columns', [])
            
            select_clause = ", ".join([f'meta."{col}"' for col in retrieval_cols])
            
            params = []

            # Dynamically build multi-column similarity calculation
            embedding_columns_config = emb_cfg['embedding_column']
            if isinstance(embedding_columns_config, list):
                similarity_expressions = [f"list_cosine_similarity(emb.\"{col}\", ?)" for col in embedding_columns_config]
                similarity_clause = f"GREATEST({', '.join(similarity_expressions)})"
                params.extend([query_embedding] * len(embedding_columns_config))
                # Add IS NOT NULL check for each column in multi-column embeddings
                not_null_conditions = [f"emb.\"{col}\" IS NOT NULL" for col in embedding_columns_config]
            else:
                similarity_clause = f"list_cosine_similarity(emb.\"{embedding_columns_config}\", ?)"
                params.append(query_embedding)
                # Add IS NOT NULL check for single-column embedding
                not_null_conditions = [f"emb.\"{embedding_columns_config}\" IS NOT NULL"]

            # Dynamically build WHERE clause
            where_conditions = []
            # Add not_null_conditions to the main where conditions list
            where_conditions.extend(not_null_conditions)

            if not is_global_source:
                where_conditions.append("meta.project_id = ?")
                params.append(project_id)

            if tags and meta_cfg.get('tags_column'):
                where_conditions.append(f"array_has_any(meta.{meta_cfg['tags_column']}, ?)")
                params.append(tags)
            
            # Assemble the final SQL query
            sql_parts = [
                f"SELECT {select_clause}, {similarity_clause} AS similarity",
                f"FROM {emb_cfg['name']} AS emb",
                f"JOIN {meta_cfg['name']} AS meta ON emb.{emb_cfg['id_column']} = meta.{meta_cfg['id_column']}"
            ]
            if where_conditions:
                sql_parts.append("WHERE " + " AND ".join(where_conditions))
            
            sql_parts.append(f"ORDER BY similarity DESC LIMIT ?")
            params.append(top_k)
            
            final_sql_query = " ".join(sql_parts)
            
            logger.debug("executing_rag_search_query", extra={"query": final_sql_query})
            logger.debug("rag_search_query_parameters", extra={"params_count": len(params)})

            with self._get_connection() as con:
                results_cursor = con.execute(final_sql_query, params)
                column_names = [desc[0] for desc in results_cursor.description]
                return [dict(zip(column_names, row)) for row in results_cursor.fetchall()]

        return await self._execute_in_thread(_sync_search)

    async def close_connection(self):
        """DuckDB connections are lightweight and usually do not need to be closed manually. This method is mainly for explicit cleanup."""
        logger.info("requesting_connection_close", extra={"database_file": self.db_file})
        # Since we do not persist the connection in __init__, there is no object to close here.
        # Connections are created and destroyed within each synchronous function block.
        pass