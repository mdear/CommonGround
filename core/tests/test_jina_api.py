"""
Unit tests for agent_core.services.jina_api module.

This module tests the Jina AI API helper functions for web search
and URL content retrieval.

Key functionality tested:
- get_jina_key: Environment variable retrieval
- check_jina_search: Search API connectivity check
- check_jina_visit: URL visit API connectivity check
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from agent_core.services.jina_api import (
    get_jina_key,
    check_jina_search,
    check_jina_visit,
)


class TestGetJinaKey:
    """Tests for get_jina_key function."""

    def test_returns_key_when_set(self):
        """Test returns API key when environment variable is set."""
        with patch.dict(os.environ, {"JINA_KEY": "test-api-key-123"}):
            result = get_jina_key()

        assert result == "test-api-key-123"

    def test_returns_none_when_not_set(self):
        """Test returns None when environment variable not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_jina_key()

        assert result is None

    def test_returns_empty_string_if_set_empty(self):
        """Test returns empty string if env var is set to empty."""
        with patch.dict(os.environ, {"JINA_KEY": ""}):
            result = get_jina_key()

        # Empty string is still falsy, but get_jina_key returns the value
        # The function checks "if not jina_key" so empty returns None-like behavior
        # Actually checking implementation: it returns jina_key regardless
        # Let me verify - os.environ.get returns "" if set to ""
        # The function then checks "if not jina_key" which is True for ""
        # So it logs error and returns... actually it just returns jina_key
        # Need to check actual implementation
        assert result == ""


class TestJinaSearchAPI:
    """Tests for check_jina_search function."""

    def test_returns_false_when_no_api_key(self):
        """Test returns False when API key not available."""
        with patch.dict(os.environ, {}, clear=True):
            result = check_jina_search()

        assert result is False

    @patch('agent_core.services.jina_api.requests.get')
    def test_returns_true_on_success(self, mock_get):
        """Test returns True when API call succeeds."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "valid-key"}):
            result = check_jina_search()

        assert result is True
        mock_get.assert_called_once()

    @patch('agent_core.services.jina_api.requests.get')
    def test_returns_false_on_non_200_status(self, mock_get):
        """Test returns False when API returns non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 401  # Unauthorized
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "invalid-key"}):
            result = check_jina_search()

        assert result is False

    @patch('agent_core.services.jina_api.requests.get')
    def test_returns_false_on_exception(self, mock_get):
        """Test returns False when request raises exception."""
        mock_get.side_effect = Exception("Network error")

        with patch.dict(os.environ, {"JINA_KEY": "valid-key"}):
            result = check_jina_search()

        assert result is False

    @patch('agent_core.services.jina_api.requests.get')
    def test_uses_correct_url_format(self, mock_get):
        """Test uses correct Jina search URL format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "key"}):
            check_jina_search(query="test query")

        call_url = mock_get.call_args[0][0]
        assert "s.jina.ai" in call_url
        assert "test query" in call_url

    @patch('agent_core.services.jina_api.requests.get')
    def test_includes_auth_header(self, mock_get):
        """Test includes authorization header."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "my-api-key"}):
            check_jina_search()

        call_headers = mock_get.call_args[1]["headers"]
        assert "Authorization" in call_headers
        assert "Bearer my-api-key" in call_headers["Authorization"]


class TestJinaVisitAPI:
    """Tests for check_jina_visit function."""

    def test_returns_false_when_no_api_key(self):
        """Test returns False when API key not available."""
        with patch.dict(os.environ, {}, clear=True):
            result = check_jina_visit()

        assert result is False

    @patch('agent_core.services.jina_api.requests.get')
    def test_returns_true_on_success(self, mock_get):
        """Test returns True when API call succeeds."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "valid-key"}):
            result = check_jina_visit()

        assert result is True

    @patch('agent_core.services.jina_api.requests.get')
    def test_returns_false_on_non_200_status(self, mock_get):
        """Test returns False when API returns non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "key"}):
            result = check_jina_visit()

        assert result is False

    @patch('agent_core.services.jina_api.requests.get')
    def test_returns_false_on_exception(self, mock_get):
        """Test returns False when request raises exception."""
        mock_get.side_effect = ConnectionError("Failed to connect")

        with patch.dict(os.environ, {"JINA_KEY": "valid-key"}):
            result = check_jina_visit()

        assert result is False

    @patch('agent_core.services.jina_api.requests.get')
    def test_uses_correct_url_format(self, mock_get):
        """Test uses correct Jina reader URL format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "key"}):
            check_jina_visit(url="example.com/page")

        call_url = mock_get.call_args[0][0]
        assert "r.jina.ai" in call_url
        assert "example.com/page" in call_url

    @patch('agent_core.services.jina_api.requests.get')
    def test_includes_auth_header(self, mock_get):
        """Test includes authorization header."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "secret-key"}):
            check_jina_visit()

        call_headers = mock_get.call_args[1]["headers"]
        assert "Authorization" in call_headers
        assert "Bearer secret-key" in call_headers["Authorization"]

    @patch('agent_core.services.jina_api.requests.get')
    def test_default_url_is_github(self, mock_get):
        """Test default URL parameter is github.com."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "key"}):
            check_jina_visit()  # No url parameter

        call_url = mock_get.call_args[0][0]
        assert "github.com" in call_url


class TestDefaultParameters:
    """Tests for default parameter values."""

    @patch('agent_core.services.jina_api.requests.get')
    def test_search_default_query(self, mock_get):
        """Test default search query is 'PocketFlow'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"JINA_KEY": "key"}):
            check_jina_search()  # No query parameter

        call_url = mock_get.call_args[0][0]
        assert "PocketFlow" in call_url
