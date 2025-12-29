"""
Unit tests for agent_core/rag/embedding_utils.py

Tests embedding providers, quantization functions, and utility functions.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import os

from agent_core.rag.embedding_utils import (
    fast_8bit_uniform_scalar_quantize,
    fast_4bit_uniform_scalar_quantize,
    l2_normalize_numpy_pytorch_like,
    EmbeddingProvider,
    LocalModelProvider,
    JinaAPIProvider,
    get_embedding_provider,
    _PROVIDER_CACHE,
    _DEFAULT_MODEL_PATH,
)


class TestFast8BitUniformScalarQuantize:
    """Tests for 8-bit quantization function."""

    def test_basic_quantization(self):
        """Test basic 8-bit quantization on simple input."""
        emb = np.array([[0.0, 0.5, -0.5]], dtype=np.float32)
        limit = 1.0

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        assert result.dtype == np.uint8
        assert result.shape == emb.shape
        # 0.0 should map to middle (127 or 128)
        assert 125 <= result[0, 0] <= 130  # Approximately middle

    def test_quantization_range(self):
        """Test that output stays in valid uint8 range."""
        # Values well within limits
        emb = np.array([[0.9, -0.9, 0.1, -0.1]], dtype=np.float32)
        limit = 1.0

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        assert result.min() >= 0
        assert result.max() <= 255

    def test_values_at_limits(self):
        """Test values at the limit boundaries."""
        limit = 1.0
        emb = np.array([[-1.0, 1.0]], dtype=np.float32)

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        # -limit should map to 0, +limit should map to 255
        assert result[0, 0] == 0  # -1.0 at limit
        assert result[0, 1] == 255  # +1.0 at limit

    def test_values_beyond_limits_clipped(self):
        """Test that values beyond limits are clipped."""
        limit = 0.5
        emb = np.array([[-1.0, 1.0]], dtype=np.float32)  # Beyond limit

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        # Should be clipped to 0 and 255
        assert result[0, 0] == 0
        assert result[0, 1] == 255

    def test_batch_processing(self):
        """Test quantization handles batches correctly."""
        emb = np.random.randn(100, 128).astype(np.float32) * 0.5
        limit = 1.0

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        assert result.shape == (100, 128)
        assert result.dtype == np.uint8


class TestFast4BitUniformScalarQuantize:
    """Tests for 4-bit quantization function."""

    def test_basic_quantization_packs_pairs(self):
        """Test that 4-bit quantization packs pairs of values."""
        # Must have even number of columns
        emb = np.array([[0.0, 0.5, -0.5, 0.25]], dtype=np.float32)
        limit = 1.0

        result = fast_4bit_uniform_scalar_quantize(emb, limit)

        assert result.dtype == np.uint8
        # Output should have half the columns (pairs packed)
        assert result.shape == (1, 2)

    def test_output_shape(self):
        """Test output shape is half the input columns."""
        emb = np.random.randn(10, 256).astype(np.float32) * 0.5
        limit = 1.0

        result = fast_4bit_uniform_scalar_quantize(emb, limit)

        assert result.shape == (10, 128)

    def test_requires_even_columns(self):
        """Test that odd column count raises assertion error."""
        emb = np.array([[0.0, 0.5, -0.5]], dtype=np.float32)  # 3 columns (odd)
        limit = 1.0

        with pytest.raises(AssertionError):
            fast_4bit_uniform_scalar_quantize(emb, limit)

    def test_quantization_range(self):
        """Test that output stays in valid uint8 range."""
        emb = np.random.randn(50, 128).astype(np.float32) * 0.5
        limit = 1.0

        result = fast_4bit_uniform_scalar_quantize(emb, limit)

        assert result.min() >= 0
        assert result.max() <= 255


class TestL2NormalizeNumpyPytorchLike:
    """Tests for L2 normalization function."""

    def test_basic_normalization(self):
        """Test that vectors are normalized to unit length."""
        arr = np.array([[3.0, 4.0]], dtype=np.float32)  # Length 5

        result = l2_normalize_numpy_pytorch_like(arr, axis=1)

        # Check unit length
        norm = np.linalg.norm(result, axis=1)
        np.testing.assert_almost_equal(norm, [1.0])
        # Check values: 3/5=0.6, 4/5=0.8
        np.testing.assert_almost_equal(result[0], [0.6, 0.8])

    def test_batch_normalization(self):
        """Test normalization of multiple vectors."""
        arr = np.array([
            [3.0, 4.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ], dtype=np.float32)

        result = l2_normalize_numpy_pytorch_like(arr, axis=1)

        # All vectors should have unit length
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_almost_equal(norms, [1.0, 1.0, 1.0])

    def test_handles_zero_vector_with_epsilon(self):
        """Test that zero vectors don't cause division by zero."""
        arr = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        epsilon = 1e-12

        result = l2_normalize_numpy_pytorch_like(arr, axis=1, epsilon=epsilon)

        # Should not raise, output should be valid
        assert result.shape == arr.shape
        assert np.isfinite(result).all()

    def test_raises_for_non_ndarray(self):
        """Test that non-ndarray input raises TypeError."""
        with pytest.raises(TypeError, match="must be a NumPy ndarray"):
            l2_normalize_numpy_pytorch_like([1, 2, 3], axis=1)

    def test_raises_for_none_axis(self):
        """Test that None axis raises ValueError."""
        arr = np.array([[1.0, 2.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="axis.*must be specified"):
            l2_normalize_numpy_pytorch_like(arr, axis=None)

    def test_handles_integer_input(self):
        """Test that integer arrays are converted to float."""
        arr = np.array([[3, 4]], dtype=np.int32)

        result = l2_normalize_numpy_pytorch_like(arr, axis=1)

        assert result.dtype in [np.float32, np.float64]
        np.testing.assert_almost_equal(result[0], [0.6, 0.8])

    def test_high_dimensional_vectors(self):
        """Test normalization of high-dimensional vectors."""
        arr = np.random.randn(10, 768).astype(np.float32)

        result = l2_normalize_numpy_pytorch_like(arr, axis=1)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_almost_equal(norms, np.ones(10), decimal=5)


class TestLocalModelProvider:
    """Tests for LocalModelProvider class."""

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_initialization_with_model_id(self, mock_text_embedding):
        """Test provider initialization with model ID."""
        provider = LocalModelProvider("test-model")

        mock_text_embedding.assert_called_once_with(model_name_or_path="test-model")

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_initialization_with_config(self, mock_text_embedding):
        """Test provider initialization with config."""
        config = {"device": "cpu", "batch_size": 32}

        provider = LocalModelProvider("test-model", config)

        mock_text_embedding.assert_called_once_with(
            model_name_or_path="test-model",
            model_config={"device": "cpu", "batch_size": 32}
        )

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_filters_api_specific_keys_from_config(self, mock_text_embedding):
        """Test that API-specific keys are filtered from config."""
        config = {
            "device": "cpu",
            "api_model_name": "should-be-removed",
            "api_key_env_var": "should-be-removed"
        }

        provider = LocalModelProvider("test-model", config)

        # Only non-API keys should be passed
        call_args = mock_text_embedding.call_args
        if call_args.kwargs.get("model_config"):
            assert "api_model_name" not in call_args.kwargs["model_config"]
            assert "api_key_env_var" not in call_args.kwargs["model_config"]

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_generate_embedding_empty_texts(self, mock_text_embedding):
        """Test generate_embedding with empty list."""
        provider = LocalModelProvider("test-model")

        result = provider.generate_embedding([])

        assert isinstance(result, np.ndarray)
        assert result.size == 0

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_generate_embedding_calls_encode(self, mock_text_embedding):
        """Test that generate_embedding calls model.encode."""
        mock_model = MagicMock()
        mock_model.model_name_or_path = "test-model"
        # Return mock embeddings
        mock_model.encode.return_value = np.random.randn(2, 256).astype(np.float32)
        mock_text_embedding.return_value = mock_model

        provider = LocalModelProvider("test-model")
        result = provider.generate_embedding(["hello", "world"])

        mock_model.encode.assert_called_once()

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_snowflake_model_prepends_query_prefix(self, mock_text_embedding):
        """Test that Snowflake models prepend 'query: ' for query task."""
        mock_model = MagicMock()
        mock_model.model_name_or_path = "Snowflake/snowflake-arctic-embed-m-v2.0"
        mock_model.encode.return_value = np.random.randn(1, 256).astype(np.float32)
        mock_text_embedding.return_value = mock_model

        provider = LocalModelProvider("Snowflake/snowflake-arctic-embed-m-v2.0")
        provider.generate_embedding(["test text"], task_type="query")

        # Check that the text was prefixed
        call_args = mock_model.encode.call_args
        assert "query: test text" in call_args[0][0]

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_mrl_truncates_dimensions(self, mock_text_embedding):
        """Test that MRL parameter truncates embedding dimensions."""
        mock_model = MagicMock()
        mock_model.model_name_or_path = "test-model"
        # Return 768-dim embeddings
        mock_model.encode.return_value = np.random.randn(2, 768).astype(np.float32)
        mock_text_embedding.return_value = mock_model

        provider = LocalModelProvider("test-model")
        result = provider.generate_embedding(["hello", "world"], mrl=128)

        # Should be truncated to 128 dims
        assert result.shape == (2, 128)

    @patch("agent_core.rag.embedding_utils.TextEmbedding")
    def test_mrl_none_returns_full_dimensions(self, mock_text_embedding):
        """Test that MRL=None returns full dimensions."""
        mock_model = MagicMock()
        mock_model.model_name_or_path = "test-model"
        mock_model.encode.return_value = np.random.randn(2, 768).astype(np.float32)
        mock_text_embedding.return_value = mock_model

        provider = LocalModelProvider("test-model")
        result = provider.generate_embedding(["hello", "world"], mrl=None)

        assert result.shape == (2, 768)


class TestJinaAPIProvider:
    """Tests for JinaAPIProvider class."""

    def test_initialization_defaults(self):
        """Test provider uses defaults when no config."""
        provider = JinaAPIProvider()

        assert provider.api_model_name == "jina-embeddings-v3"
        assert provider.api_key_env_var == "JINA_KEY"
        assert provider.api_url == 'https://api.jina.ai/v1/embeddings'

    def test_initialization_with_config(self):
        """Test provider uses config values."""
        config = {
            "api_model_name": "custom-model",
            "api_key_env_var": "CUSTOM_KEY"
        }

        provider = JinaAPIProvider(config)

        assert provider.api_model_name == "custom-model"
        assert provider.api_key_env_var == "CUSTOM_KEY"

    def test_generate_embedding_empty_texts(self):
        """Test generate_embedding with empty list."""
        provider = JinaAPIProvider()

        result = provider.generate_embedding([])

        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_generate_embedding_raises_without_api_key(self):
        """Test generate_embedding raises when API key not set."""
        provider = JinaAPIProvider({"api_key_env_var": "NONEXISTENT_KEY"})

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NONEXISTENT_KEY", None)

            with pytest.raises(ValueError, match="API key not found"):
                provider.generate_embedding(["test"])

    @patch("agent_core.rag.embedding_utils.requests.post")
    def test_generate_embedding_makes_api_call(self, mock_post):
        """Test that generate_embedding makes correct API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3] * 50}  # 150 dims
            ]
        }
        mock_post.return_value = mock_response

        provider = JinaAPIProvider()

        with patch.dict(os.environ, {"JINA_KEY": "test-key"}):
            result = provider.generate_embedding(["hello"], mrl=128)

        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL
        assert call_args[0][0] == 'https://api.jina.ai/v1/embeddings'
        # Check headers
        assert "Authorization" in call_args[1]["headers"]
        assert "Bearer test-key" in call_args[1]["headers"]["Authorization"]
        # Check payload
        assert call_args[1]["json"]["input"] == ["hello"]

    @patch("agent_core.rag.embedding_utils.requests.post")
    def test_generate_embedding_includes_task(self, mock_post):
        """Test that task_type is included in API request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 256}]
        }
        mock_post.return_value = mock_response

        provider = JinaAPIProvider()

        with patch.dict(os.environ, {"JINA_KEY": "test-key"}):
            provider.generate_embedding(["hello"], task_type="retrieval.query")

        call_args = mock_post.call_args
        assert call_args[1]["json"]["task"] == "retrieval.query"

    @patch("agent_core.rag.embedding_utils.requests.post")
    def test_generate_embedding_handles_request_error(self, mock_post):
        """Test graceful handling of request errors."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

        provider = JinaAPIProvider()

        with patch.dict(os.environ, {"JINA_KEY": "test-key"}):
            result = provider.generate_embedding(["hello"])

        # Should return empty array on error
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    @patch("agent_core.rag.embedding_utils.requests.post")
    def test_generate_embedding_handles_malformed_response(self, mock_post):
        """Test handling of malformed API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "invalid"}  # Missing 'data'
        mock_post.return_value = mock_response

        provider = JinaAPIProvider()

        with patch.dict(os.environ, {"JINA_KEY": "test-key"}):
            result = provider.generate_embedding(["hello"])

        # Should return empty array on malformed response
        assert result.size == 0


class TestGetEmbeddingProvider:
    """Tests for the provider factory function."""

    def setup_method(self):
        """Clear provider cache before each test."""
        _PROVIDER_CACHE.clear()

    @patch("agent_core.rag.embedding_utils.LocalModelProvider")
    def test_returns_local_provider_for_model_path(self, mock_local_provider):
        """Test that model paths return LocalModelProvider."""
        mock_instance = MagicMock()
        mock_local_provider.return_value = mock_instance

        provider = get_embedding_provider("Snowflake/snowflake-arctic-embed-m-v2.0")

        mock_local_provider.assert_called_once_with(
            "Snowflake/snowflake-arctic-embed-m-v2.0", None
        )

    @patch("agent_core.rag.embedding_utils.JinaAPIProvider")
    def test_returns_jina_provider_for_jina_api(self, mock_jina_provider):
        """Test that 'jina-api' returns JinaAPIProvider."""
        mock_instance = MagicMock()
        mock_jina_provider.return_value = mock_instance

        provider = get_embedding_provider("jina-api")

        mock_jina_provider.assert_called_once_with(None)

    @patch("agent_core.rag.embedding_utils.LocalModelProvider")
    def test_uses_default_model_when_none(self, mock_local_provider):
        """Test that None model_id uses default."""
        mock_instance = MagicMock()
        mock_local_provider.return_value = mock_instance

        provider = get_embedding_provider(None)

        mock_local_provider.assert_called_once_with(_DEFAULT_MODEL_PATH, None)

    @patch("agent_core.rag.embedding_utils.LocalModelProvider")
    def test_caches_providers(self, mock_local_provider):
        """Test that providers are cached."""
        mock_instance = MagicMock()
        mock_local_provider.return_value = mock_instance

        provider1 = get_embedding_provider("test-model")
        provider2 = get_embedding_provider("test-model")

        # Should only create once
        assert mock_local_provider.call_count == 1
        assert provider1 is provider2

    @patch("agent_core.rag.embedding_utils.LocalModelProvider")
    def test_cache_key_includes_config(self, mock_local_provider):
        """Test that different configs create different cache entries."""
        mock_local_provider.side_effect = [MagicMock(), MagicMock()]

        provider1 = get_embedding_provider("test-model", {"batch_size": 16})
        provider2 = get_embedding_provider("test-model", {"batch_size": 32})

        # Should create two different providers
        assert mock_local_provider.call_count == 2
        assert provider1 is not provider2

    @patch("agent_core.rag.embedding_utils.LocalModelProvider")
    def test_passes_config_to_provider(self, mock_local_provider):
        """Test that model_config is passed to provider."""
        mock_instance = MagicMock()
        mock_local_provider.return_value = mock_instance
        config = {"device": "cuda"}

        provider = get_embedding_provider("test-model", config)

        mock_local_provider.assert_called_once_with("test-model", config)


class TestEmbeddingProviderAbstract:
    """Tests for EmbeddingProvider abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_subclass_must_implement_generate_embedding(self):
        """Test that subclasses must implement generate_embedding."""
        class IncompleteProvider(EmbeddingProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_subclass_with_implementation_works(self):
        """Test that complete subclass can be instantiated."""
        class CompleteProvider(EmbeddingProvider):
            def generate_embedding(self, texts, task_type="", mrl=128):
                return np.array([])

        provider = CompleteProvider()
        assert isinstance(provider, EmbeddingProvider)


class TestQuantizationEdgeCases:
    """Edge case tests for quantization functions."""

    def test_8bit_single_value(self):
        """Test 8-bit quantization with single value."""
        emb = np.array([[0.0]], dtype=np.float32)
        limit = 1.0

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        assert result.shape == (1, 1)

    def test_8bit_very_small_limit(self):
        """Test 8-bit quantization with very small limit."""
        emb = np.array([[0.0, 0.001, -0.001]], dtype=np.float32)
        limit = 0.01

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        assert result.dtype == np.uint8

    def test_4bit_minimum_valid_size(self):
        """Test 4-bit quantization with minimum valid size (2 columns)."""
        emb = np.array([[0.0, 0.5]], dtype=np.float32)
        limit = 1.0

        result = fast_4bit_uniform_scalar_quantize(emb, limit)

        assert result.shape == (1, 1)

    def test_large_batch_8bit(self):
        """Test 8-bit quantization with large batch."""
        emb = np.random.randn(1000, 512).astype(np.float32) * 0.3
        limit = 1.0

        result = fast_8bit_uniform_scalar_quantize(emb, limit)

        assert result.shape == (1000, 512)
        assert result.dtype == np.uint8

    def test_large_batch_4bit(self):
        """Test 4-bit quantization with large batch."""
        emb = np.random.randn(1000, 512).astype(np.float32) * 0.3
        limit = 1.0

        result = fast_4bit_uniform_scalar_quantize(emb, limit)

        assert result.shape == (1000, 256)
        assert result.dtype == np.uint8
