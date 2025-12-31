import os
import logging
import requests

# Configure logging
logger = logging.getLogger(__name__)

def get_jina_key():
    """
    Get Jina API key from environment variables.
    
    Returns:
        str: The Jina API key, or None if not set.
    """
    jina_key = os.environ.get('JINA_KEY')
    if not jina_key:
        logger.error("JINA_KEY environment variable is not set")
    return jina_key

def check_jina_search(query="PocketFlow"):
    """
    Check the Jina Search API connection.
    
    Args:
        query (str, optional): The test search query. Defaults to "PocketFlow".
    
    Returns:
        bool: Whether the connection check was successful.
    """
    jina_key = get_jina_key()
    if not jina_key:
        return False
    
    try:
        url = f'https://s.jina.ai/?q={query}'
        headers = {
            'Authorization': f'Bearer {jina_key}',
            'X-Respond-With': 'no-content'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error("jina_search_api_test_failed", extra={"status_code": response.status_code})
            return False
        
        logger.info("Jina Search API test successful")
        return True
    
    except Exception as e:
        logger.error("jina_search_api_test_error", extra={"error_message": str(e)})
        return False

def check_jina_visit(url="github.com"):
    """
    Check the Jina URL Visit API connection.
    
    Args:
        url (str, optional): The test URL to visit. Defaults to "github.com".
    
    Returns:
        bool: Whether the connection check was successful.
    """
    jina_key = get_jina_key()
    if not jina_key:
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {jina_key}"
        }
        
        response = requests.get(f'https://r.jina.ai/{url}', headers=headers)
        
        if response.status_code != 200:
            logger.error("jina_visit_api_test_failed", extra={"status_code": response.status_code})
            return False
        
        logger.info("Jina URL Visit API test successful")
        return True
    
    except Exception as e:
        logger.error("jina_visit_api_test_error", extra={"error_message": str(e)})
        return False

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Test Jina API
    print("Testing Jina Search API:", "Success" if check_jina_search() else "Failure")
    print("Testing Jina Visit API:", "Success" if check_jina_visit() else "Failure")
