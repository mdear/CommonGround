"""
Tier 3 Unit Tests for agent_core/rag/search_engine.py

Tests the RAG search engine functionality:
- Database connection management with caching
- Search configuration loading and validation
- Vector similarity search
- Direct metadata search

Test Categories:
1. Database Connections: get_db_connection() caching and upgrade logic
2. Config Loading: load_search_config(), get_search_config_details()
3. Vector Search: search_similar_documents()
4. Direct Search: direct_search()
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import yaml
import os
import tempfile

from agent_core.rag.search_engine import (
    get_db_connection,
    load_search_config,
    get_search_config_details,
    search_similar_documents,
    direct_search,
    _DB_CONNECTION_CACHE,
    SCALAR_QUANTIZATION_LIMIT_8BIT,
    SCALAR_QUANTIZATION_LIMIT_4BIT
)


class TestGetDbConnection:
    """Tests for the get_db_connection function."""

    def setup_method(self):
        """Clear the connection cache before each test."""
        _DB_CONNECTION_CACHE.clear()

    def teardown_method(self):
        """Clean up connections after each test."""
        for db_file, (conn, _) in list(_DB_CONNECTION_CACHE.items()):
            try:
                conn.close()
            except:
                pass
        _DB_CONNECTION_CACHE.clear()

    @patch('agent_core.rag.search_engine.duckdb.connect')
    def test_creates_new_connection(self, mock_connect):
        """Test that a new connection is created when not cached."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        result = get_db_connection("/path/to/db.duckdb", read_only=True)

        mock_connect.assert_called_once_with(database="/path/to/db.duckdb", read_only=True)
        assert result is mock_conn
        assert "/path/to/db.duckdb" in _DB_CONNECTION_CACHE

    @patch('agent_core.rag.search_engine.duckdb.connect')
    def test_returns_cached_connection(self, mock_connect):
        """Test that cached connections are returned."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # First call creates connection
        conn1 = get_db_connection("/path/to/cached.db", read_only=True)
        # Second call should return cached
        conn2 = get_db_connection("/path/to/cached.db", read_only=True)

        assert mock_connect.call_count == 1
        assert conn1 is conn2

    @patch('agent_core.rag.search_engine.duckdb.connect')
    def test_upgrades_readonly_to_writable(self, mock_connect):
        """Test that read-only connections are upgraded when write access is needed."""
        mock_ro_conn = MagicMock()
        mock_rw_conn = MagicMock()
        mock_connect.side_effect = [mock_ro_conn, mock_rw_conn]

        # First, get a read-only connection
        conn1 = get_db_connection("/path/to/upgrade.db", read_only=True)
        assert conn1 is mock_ro_conn

        # Now request a writable connection
        conn2 = get_db_connection("/path/to/upgrade.db", read_only=False)

        # Should have closed the old connection and created a new one
        mock_ro_conn.close.assert_called_once()
        assert conn2 is mock_rw_conn
        assert mock_connect.call_count == 2

    @patch('agent_core.rag.search_engine.duckdb.connect')
    def test_no_upgrade_needed_if_already_writable(self, mock_connect):
        """Test that writable connections satisfy read-only requests."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Get a writable connection first
        get_db_connection("/path/to/writable.db", read_only=False)
        # Request read-only - should reuse writable
        conn2 = get_db_connection("/path/to/writable.db", read_only=True)

        assert mock_connect.call_count == 1  # No second connection created
        assert conn2 is mock_conn

    @patch('agent_core.rag.search_engine.duckdb.connect')
    def test_connection_error_raised(self, mock_connect):
        """Test that connection errors are propagated."""
        import duckdb
        mock_connect.side_effect = duckdb.Error("Connection failed")

        with pytest.raises(duckdb.Error, match="Connection failed"):
            get_db_connection("/path/to/bad.db", read_only=True)


class TestLoadSearchConfig:
    """Tests for the load_search_config function."""

    def test_loads_valid_yaml(self, tmp_path):
        """Test loading a valid YAML config file."""
        config_data = {
            "database_file": "test.db",
            "meta_table": {"name": "docs"},
            "embedding_table": {"name": "embeddings"}
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = load_search_config(str(config_file))

        assert result == config_data

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_search_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test that invalid YAML raises an error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: {[")

        with pytest.raises(yaml.YAMLError):
            load_search_config(str(config_file))

    def test_non_dict_yaml_raises(self, tmp_path):
        """Test that non-dictionary YAML content raises ValueError."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(ValueError, match="not a valid YAML dictionary"):
            load_search_config(str(config_file))


class TestGetSearchConfigDetails:
    """Tests for the get_search_config_details function."""

    def test_complete_config(self, tmp_path):
        """Test parsing a complete and valid config."""
        config_data = {
            "description": "Test search config",
            "database_writable": True,
            "is_global": True,
            "database_file": "test.duckdb",
            "source_name": "test_source",
            "meta_table": {
                "name": "documents",
                "id_column": "doc_id",
                "text_column": "content",
                "tags_column": "tags",
                "retrieval_columns": ["title", "author"],
                "direct_search_columns": ["category"]
            },
            "embedding_table": {
                "name": "embeddings",
                "id_column": "doc_id",
                "embedding_column": "vector",
                "emb_model_id": "text-embedding-3-small",
                "model_config_params": {"dimensions": 512},
                "quantization": "int8",
                "mrl_dims": 256,
                "query_task_type": "retrieval.query"
            }
        }
        config_file = tmp_path / "complete.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        assert result is not None
        assert result["description"] == "Test search config"
        assert result["database_writable"] is True
        assert result["is_global"] is True
        assert result["meta_table_name"] == "documents"
        assert result["meta_id_col"] == "doc_id"
        assert result["meta_text_col"] == "content"
        assert result["emb_table_name"] == "embeddings"
        assert result["emb_model_id"] == "text-embedding-3-small"
        assert result["quantization"] == "int8"
        assert result["emb_mrl_dims"] == 256
        # Database file should be resolved to absolute path
        assert os.path.isabs(result["database_file"])

    def test_relative_database_path_resolved(self, tmp_path):
        """Test that relative database paths are resolved correctly."""
        config_data = {
            "database_file": "data/mydb.duckdb",
            "meta_table": {"name": "docs", "id_column": "id"},
            "embedding_table": {
                "name": "emb",
                "id_column": "id",
                "embedding_column": "vec",
                "emb_model_id": "model"
            }
        }
        config_file = tmp_path / "relative.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        expected_db_path = os.path.join(str(tmp_path), "data/mydb.duckdb")
        assert result["database_file"] == expected_db_path

    def test_absolute_database_path_unchanged(self, tmp_path):
        """Test that absolute database paths are preserved."""
        config_data = {
            "database_file": "/absolute/path/mydb.duckdb",
            "meta_table": {"name": "docs", "id_column": "id"},
            "embedding_table": {
                "name": "emb",
                "id_column": "id",
                "embedding_column": "vec",
                "emb_model_id": "model"
            }
        }
        config_file = tmp_path / "absolute.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        assert result["database_file"] == "/absolute/path/mydb.duckdb"

    def test_missing_required_fields_returns_none(self, tmp_path):
        """Test that missing required fields cause None return."""
        # Missing embedding table
        config_data = {
            "database_file": "test.db",
            "meta_table": {"name": "docs", "id_column": "id"}
        }
        config_file = tmp_path / "incomplete.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        assert result is None

    def test_missing_meta_id_column_returns_none(self, tmp_path):
        """Test that missing meta_id_col returns None."""
        config_data = {
            "database_file": "test.db",
            "meta_table": {"name": "docs"},  # Missing id_column
            "embedding_table": {
                "name": "emb",
                "id_column": "id",
                "embedding_column": "vec",
                "emb_model_id": "model"
            }
        }
        config_file = tmp_path / "no_meta_id.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        assert result is None

    def test_missing_emb_model_id_returns_none(self, tmp_path):
        """Test that missing emb_model_id returns None."""
        config_data = {
            "database_file": "test.db",
            "meta_table": {"name": "docs", "id_column": "id"},
            "embedding_table": {
                "name": "emb",
                "id_column": "id",
                "embedding_column": "vec"
                # Missing emb_model_id
            }
        }
        config_file = tmp_path / "no_model.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        assert result is None

    def test_invalid_config_file_returns_none(self):
        """Test that invalid config files return None."""
        result = get_search_config_details("/nonexistent/config.yaml")
        assert result is None

    def test_retrieval_columns_merged_with_text_col(self, tmp_path):
        """Test that retrieval columns include text column."""
        config_data = {
            "database_file": "test.db",
            "meta_table": {
                "name": "docs",
                "id_column": "id",
                "text_column": "content",
                "retrieval_columns": ["title", "content"]  # Duplicate
            },
            "embedding_table": {
                "name": "emb",
                "id_column": "id",
                "embedding_column": "vec",
                "emb_model_id": "model"
            }
        }
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        # Should have content first (as text_col), then title (without duplicate)
        assert result["all_meta_retrieval_cols"] == ["content", "title"]

    def test_default_values(self, tmp_path):
        """Test that default values are applied."""
        config_data = {
            "database_file": "test.db",
            "meta_table": {"name": "docs", "id_column": "id"},
            "embedding_table": {
                "name": "emb",
                "id_column": "id",
                "embedding_column": "vec",
                "emb_model_id": "model"
            }
        }
        config_file = tmp_path / "defaults.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = get_search_config_details(str(config_file))

        assert result["database_writable"] is False  # Default
        assert result["is_global"] is False  # Default
        assert result["description"] == "No description provided."  # Default


class TestSearchSimilarDocuments:
    """Tests for the search_similar_documents function."""

    def test_empty_query_returns_empty(self):
        """Test that empty query returns empty results."""
        result = search_similar_documents(
            query_text="",
            search_config_details={},
            model_instance=MagicMock()
        )
        assert result == []

    def test_none_model_raises(self):
        """Test that None model_instance raises ValueError."""
        with pytest.raises(ValueError, match="EmbeddingProvider instance must be provided"):
            search_similar_documents(
                query_text="test query",
                search_config_details={
                    "database_file": "test.db",
                    "meta_table_name": "docs",
                    "meta_id_col": "id",
                    "meta_tags_col": None,
                    "all_meta_retrieval_cols": [],
                    "emb_table_name": "emb",
                    "emb_id_col": "id",
                    "emb_vector_col": "vector",
                    "emb_mrl_dims": None,
                    "query_task_type": None
                },
                model_instance=None
            )

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_embedding_generation_failure_returns_empty(self, mock_get_conn):
        """Test that embedding generation failure returns empty results."""
        mock_model = MagicMock()
        mock_model.generate_embedding.return_value = None

        result = search_similar_documents(
            query_text="test query",
            search_config_details={
                "database_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": ["content"],
                "meta_tags_col": None,
                "emb_table_name": "emb",
                "emb_id_col": "id",
                "emb_vector_col": "vector",
                "emb_mrl_dims": 256,
                "query_task_type": "retrieval.query",
                "quantization": None
            },
            model_instance=mock_model
        )

        assert result == []

    @patch('agent_core.rag.search_engine.get_db_connection')
    @patch('agent_core.rag.search_engine.fast_8bit_uniform_scalar_quantize')
    def test_int8_quantization_applied(self, mock_quantize, mock_get_conn):
        """Test that int8 quantization is applied when configured."""
        mock_model = MagicMock()
        mock_model.generate_embedding.return_value = [[0.1, 0.2, 0.3]]

        mock_quantize.return_value = MagicMock()
        mock_quantize.return_value.tolist.return_value = [[1, 2, 3]]

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",), ("content",), ("similarity",)]
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        search_similar_documents(
            query_text="test",
            search_config_details={
                "database_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": ["content"],
                "meta_tags_col": None,
                "emb_table_name": "emb",
                "emb_id_col": "id",
                "emb_vector_col": "vector",
                "emb_mrl_dims": None,
                "query_task_type": None,
                "quantization": "int8"
            },
            model_instance=mock_model
        )

        mock_quantize.assert_called_once()

    @patch('agent_core.rag.search_engine.get_db_connection')
    @patch('agent_core.rag.search_engine.fast_4bit_uniform_scalar_quantize')
    def test_int4_quantization_applied(self, mock_quantize, mock_get_conn):
        """Test that int4 quantization is applied when configured."""
        mock_model = MagicMock()
        mock_model.generate_embedding.return_value = [[0.1, 0.2, 0.3]]

        mock_quantize.return_value = MagicMock()
        mock_quantize.return_value.tolist.return_value = [[1, 2, 3]]

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",), ("similarity",)]
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        search_similar_documents(
            query_text="test",
            search_config_details={
                "database_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": [],
                "meta_tags_col": None,
                "emb_table_name": "emb",
                "emb_id_col": "id",
                "emb_vector_col": "vector",
                "emb_mrl_dims": None,
                "query_task_type": None,
                "quantization": "int4"
            },
            model_instance=mock_model
        )

        mock_quantize.assert_called_once()

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_search_with_tags_filter(self, mock_get_conn):
        """Test that tag filtering is applied in SQL."""
        import numpy as np
        mock_model = MagicMock()
        # Return numpy array to match expected format
        mock_model.generate_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",), ("similarity",)]
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        search_similar_documents(
            query_text="test",
            search_config_details={
                "database_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": [],
                "meta_tags_col": "tags",
                "emb_table_name": "emb",
                "emb_id_col": "id",
                "emb_vector_col": "vector",
                "emb_mrl_dims": None,
                "query_task_type": None,
                "quantization": None
            },
            tags=["tag1", "tag2"],
            model_instance=mock_model
        )

        # Check that SQL includes tag filter
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "array_has_all" in sql
        assert ["tag1", "tag2"] in params

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_search_returns_formatted_results(self, mock_get_conn):
        """Test that search results are properly formatted as dicts."""
        import numpy as np
        mock_model = MagicMock()
        # Return numpy array to match expected format
        mock_model.generate_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",), ("content",), ("similarity",)]
        mock_result.fetchall.return_value = [
            ("doc1", "Test content 1", 0.95),
            ("doc2", "Test content 2", 0.85)
        ]
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        results = search_similar_documents(
            query_text="test",
            search_config_details={
                "database_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": ["content"],
                "meta_tags_col": None,
                "emb_table_name": "emb",
                "emb_id_col": "id",
                "emb_vector_col": "vector",
                "emb_mrl_dims": None,
                "query_task_type": None,
                "quantization": None
            },
            top_k=10,
            model_instance=mock_model
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "Test content 1"
        assert results[0]["similarity"] == 0.95

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_multiple_embedding_columns(self, mock_get_conn):
        """Test search with multiple embedding columns uses GREATEST."""
        import numpy as np
        mock_model = MagicMock()
        # Return numpy array to match expected format
        mock_model.generate_embedding.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",), ("similarity",)]
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        search_similar_documents(
            query_text="test",
            search_config_details={
                "database_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": [],
                "meta_tags_col": None,
                "emb_table_name": "emb",
                "emb_id_col": "id",
                "emb_vector_col": ["vec_title", "vec_content"],  # Multiple columns
                "emb_mrl_dims": None,
                "query_task_type": None,
                "quantization": None
            },
            model_instance=mock_model
        )

        sql = mock_conn.execute.call_args[0][0]
        assert "GREATEST" in sql
        assert "vec_title" in sql
        assert "vec_content" in sql


class TestDirectSearch:
    """Tests for the direct_search function."""

    def test_empty_query_params_returns_empty(self):
        """Test that empty query params returns empty results."""
        result = direct_search(
            search_config_details={
                "db_file": "test.db",
                "meta_table_name": "docs",
                "all_meta_retrieval_cols": [],
                "meta_id_col": "id"
            },
            query_params={}
        )
        assert result == []

    def test_disallowed_column_returns_empty(self):
        """Test that querying disallowed columns returns empty."""
        result = direct_search(
            search_config_details={
                "db_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": [],
                "direct_search_columns": ["allowed_col"]
            },
            query_params={"disallowed_col": "value"}
        )
        assert result == []

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_valid_search_executes_sql(self, mock_get_conn):
        """Test that valid searches execute proper SQL."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",), ("name",)]
        mock_result.fetchall.return_value = [("1", "Doc 1"), ("2", "Doc 2")]
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        results = direct_search(
            search_config_details={
                "db_file": "test.db",
                "meta_table_name": "documents",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": ["name"],
                "direct_search_columns": ["category", "author"]
            },
            query_params={"category": "tech"},
            limit=5
        )

        assert len(results) == 2
        # Check SQL structure
        sql = mock_conn.execute.call_args[0][0]
        params = mock_conn.execute.call_args[0][1]
        assert '"category" = ?' in sql
        assert "tech" in params
        assert 5 in params  # limit

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_multiple_query_params(self, mock_get_conn):
        """Test search with multiple query parameters."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",)]
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        direct_search(
            search_config_details={
                "db_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": [],
                "direct_search_columns": ["category", "status"]
            },
            query_params={"category": "tech", "status": "active"}
        )

        sql = mock_conn.execute.call_args[0][0]
        assert '"category" = ?' in sql
        assert '"status" = ?' in sql
        assert " AND " in sql

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_results_formatted_as_dicts(self, mock_get_conn):
        """Test that results are returned as list of dicts."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.description = [("id",), ("title",), ("content",)]
        mock_result.fetchall.return_value = [
            ("doc1", "Title 1", "Content 1"),
            ("doc2", "Title 2", "Content 2")
        ]
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value = mock_conn

        results = direct_search(
            search_config_details={
                "db_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": ["title", "content"],
                "direct_search_columns": ["category"]
            },
            query_params={"category": "test"}
        )

        assert results[0] == {"id": "doc1", "title": "Title 1", "content": "Content 1"}
        assert results[1] == {"id": "doc2", "title": "Title 2", "content": "Content 2"}

    @patch('agent_core.rag.search_engine.get_db_connection')
    def test_duckdb_error_returns_empty(self, mock_get_conn):
        """Test that DuckDB errors are handled gracefully."""
        import duckdb
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = duckdb.Error("Query failed")
        mock_get_conn.return_value = mock_conn

        result = direct_search(
            search_config_details={
                "db_file": "test.db",
                "meta_table_name": "docs",
                "meta_id_col": "id",
                "all_meta_retrieval_cols": [],
                "direct_search_columns": ["col"]
            },
            query_params={"col": "val"}
        )

        assert result == []


class TestScalarQuantizationLimits:
    """Tests for scalar quantization limit constants."""

    def test_8bit_limit_value(self):
        """Test that 8-bit limit is correctly defined."""
        assert SCALAR_QUANTIZATION_LIMIT_8BIT == 0.3

    def test_4bit_limit_value(self):
        """Test that 4-bit limit is correctly defined."""
        assert SCALAR_QUANTIZATION_LIMIT_4BIT == 0.18
