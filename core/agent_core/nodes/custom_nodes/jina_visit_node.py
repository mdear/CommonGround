import logging
import asyncio
import aiohttp
from ..base_tool_node import BaseToolNode
from ...services.jina_api import get_jina_key

import os

COMPRESS_WEB_CONTENT_TO_LENGTH = os.getenv("COMPRESS_WEB_CONTENT_TO_LENGTH", "")

# Configure logging
logger = logging.getLogger(__name__)

from ...framework.tool_registry import tool_registry # <--- Add import

@tool_registry(
    name="visit_url",
    description="Visits a given URL or a list of URLs and returns their content. This tool is used to fetch the content of specific webpages.",
    parameters={
        "type": "object",
        "properties": {
            "url": { # Single URL
                "type": "string",
                "description": "The single URL to visit."
            },
            "urls": { # List of URLs
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of URLs to visit. Use either 'url' or 'urls'."
            },
            "purpose": { 
                "type": "string",
                "description": "The purpose of visiting the URL(s). This helps in contextualizing the fetched content."
            },
            "context": { 
                "type": "string",
                "description": "Additional context or specific questions related to the content being fetched from the URL(s)."
            },
            "no_compress": { 
                "type": "boolean",
                "default": False,
                "description": "If true, attempts to skip content compression or summarization steps if they exist after fetching. Defaults to false."
            }
        },
        "required": ["purpose", "url"],
        # The LLM should provide one of 'url' or 'urls' as needed.
    },
    toolset_name="jina_search_and_visit", 
    ends_flow=False,
    default_knowledge_item_type="WEB_PAGE_CONTENT",
    source_uri_field_in_output="url", # Assuming main_content_for_llm per URL has 'url'
    title_field_in_output="title" # Assuming main_content_for_llm per URL might have 'title'
)
class JinaVisitNode(BaseToolNode):
    """
    [Refactored] Use Jina API to handle URL visits.
    All logic is encapsulated in exec_async to conform to the BaseToolNode specification.
    """

    async def _fetch_single_url(self, url: str, purpose: str, context: str):
        """Private helper method to visit a single URL and return a standard result structure."""
        logger.debug("visiting_url", extra={"url": url, "purpose": purpose, "context": context})
        
        page_content = None
        success_flag = False
        error_message = None
        extracted_title = url

        try:
            jina_key = get_jina_key()
            if not jina_key:
                error_message = "JINA_KEY environment variable is not set"
                logger.error("jina_api_key_missing", extra={"url": url})
            else:
                headers = {"Authorization": f"Bearer {jina_key}"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'https://r.jina.ai/{url}', headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            page_content = await response.text()
                            if page_content:
                                try:
                                    lower_content = page_content.lower()
                                    title_start = lower_content.find("<title>")
                                    title_end = lower_content.find("</title>", title_start)
                                    if title_start != -1 and title_end != -1:
                                        extracted_title = page_content[title_start + len("<title>"):title_end].strip()
                                    else:
                                        h1_start = lower_content.find("<h1>")
                                        h1_end = lower_content.find("</h1>", h1_start)
                                        if h1_start != -1 and h1_end != -1:
                                            extracted_title = page_content[h1_start + len("<h1>"):h1_end].strip()
                                except Exception: pass
                            success_flag = True
                        else:
                            error_text = await response.text()
                            error_message = f"Failed to visit, status code: {response.status}"
                            # Log with appropriate severity based on error type
                            if response.status == 402:
                                logger.error("jina_api_insufficient_balance", extra={
                                    "url": url,
                                    "http_status": response.status,
                                    "error_detail": error_text,
                                    "action_required": "Recharge Jina API account at https://jina.ai"
                                })
                            elif response.status == 429:
                                logger.warning("jina_api_rate_limited", extra={
                                    "url": url,
                                    "http_status": response.status,
                                    "error_detail": error_text
                                })
                            else:
                                logger.error("jina_api_error", extra={
                                    "url": url,
                                    "http_status": response.status,
                                    "error_detail": error_text
                                })
        except asyncio.TimeoutError:
            error_message = "Timeout when visiting URL"
            logger.warning("jina_api_timeout", extra={"url": url})
        except Exception as e:
            error_message = f"Error visiting URL: {str(e)}"
            logger.error("jina_api_exception", extra={"url": url, "error": str(e)})

        # Prepare content for the LLM
        main_content_for_llm = {
            "url": url,
            "success": success_flag,
            "content_preview": (page_content[:1000] + "..." if page_content and len(page_content) > 1000 else page_content) if success_flag else f"[Visit failed: {error_message}]",
            "title": extracted_title,
            "error_message": error_message if not success_flag else None
        }

        # Prepare content to be stored in the knowledge base
        knowledge_items_to_add = []
        if success_flag and page_content is not None:
            knowledge_items_to_add.append({
                "item_type": "WEB_PAGE_CONTENT",
                "content": page_content,
                "source_uri": url,
                "metadata": {
                    "title": extracted_title,
                    "original_query": context,
                    "purpose": purpose,
                }
            })
            
        return {"main_content_for_llm": main_content_for_llm, "_knowledge_items_to_add": knowledge_items_to_add}

    async def exec_async(self, prep_res: dict):
        """
        Core execution logic:
        1. Get parameters from prep_res.
        2. Check the knowledge base (KB) to separate cached URLs from those that need to be fetched live.
        3. Concurrently execute fetches for URLs that need to be retrieved in real-time.
        4. Aggregate all results and construct the final return value.
        """
        tool_params = prep_res.get("tool_params", {})
        shared_context = prep_res.get("shared_context", {})
        
        input_urls = []
        if "url" in tool_params: input_urls.append(tool_params["url"])
        if "urls" in tool_params: input_urls.extend(tool_params["urls"])
        unique_urls = sorted(list(set(filter(None, input_urls))))

        purpose = tool_params.get("purpose", "No purpose provided.")
        context_param = tool_params.get("context", "")

        kb = shared_context.get('refs', {}).get('run', {}).get('runtime', {}).get("knowledge_base")
        urls_to_fetch = []
        cached_results_for_llm = []

        if kb:
            for url in unique_urls:
                existing_item = await kb.get_item_by_uri(url)
                if existing_item:
                    logger.info("url_found_in_kb", extra={"url": url})
                    cached_results_for_llm.append({
                        "url": url, "success": True,
                        "content_preview": (str(existing_item["content"])[:1000] + "... (from knowledge base)" if existing_item.get("content") else "[Empty content, from knowledge base]"),
                        "title": existing_item.get("metadata", {}).get("title", url),
                        "source": "knowledge_base"
                    })
                else:
                    urls_to_fetch.append(url)
        else:
            urls_to_fetch = unique_urls

        live_fetch_results = []
        if urls_to_fetch:
            tasks = [self._fetch_single_url(u, purpose, context_param) for u in urls_to_fetch]
            live_fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_main_content_parts = list(cached_results_for_llm)
        all_knowledge_items_to_add = []

        for res in live_fetch_results:
            if isinstance(res, Exception):
                logger.error("parallel_jina_visit_error", extra={"error": str(res)}, exc_info=res)
                all_main_content_parts.append({"success": False, "error_message": str(res)})
            else:
                all_main_content_parts.append(res["main_content_for_llm"])
                all_knowledge_items_to_add.extend(res["_knowledge_items_to_add"])
        
        final_payload_for_llm = {
            "main_content_for_llm": all_main_content_parts
        }

        is_error = not any(part.get("success") for part in all_main_content_parts) if all_main_content_parts else True
        
        return {
            "status": "error" if is_error else "success",
            "payload": final_payload_for_llm,
            "error_message": "One or more URLs could not be visited." if is_error else None,
            "_knowledge_items_to_add": all_knowledge_items_to_add
        }

#    def get_next_step_instructions(self): # Removed
#        """Generate next step instructions based on the visit result, multilingual"""
#        if self.language == "en": 
#            return "Please analyze the above crawling results, whether you can answer the user's question. If you can, please return the answer directly. If you cannot, please use the web_search tool, continue searching based on the above crawling results to supplement the information you need, note that you should not repeat the search for URLs that have already been searched."
if __name__ == "__main__":
    # Standalone test for JinaVisitNode
    node = JinaVisitNode()
    
    # Create test state using the same shared structure as in main.py
    test_shared = {
        "state": {
            "question": "What is PocketFlow?",
            # "language": "zh", # Removed
            "searched_queries": [],
            "visited_urls": [],
            "search_history": [],
            "visit_history": [],
            "messages": [],
            "current_action": {
                "type": "visit_url",
                "url": "https://github.com/the-pocket/PocketFlow"
            },
            "current_tool_call_id": "call456"
        }
    }
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Run the node
    async def run_test():
        action = await node.run_async(test_shared)
        
        # Print the results
        print(f"Node returned action: {action}")
        print(f"Visit history: {len(test_shared['state']['visit_history'])} items")
        
        # Print the URL and status of the first visit result (if any)
        if test_shared["state"]["visit_history"]:
            first_visit = test_shared["state"]["visit_history"][0]
            print(f"First visit result: URL={first_visit.get('url')}, Success={first_visit.get('success')}")
            print(f"Content length: {len(first_visit.get('content', ''))}")
    
    # Run the async test
    asyncio.run(run_test()) 
