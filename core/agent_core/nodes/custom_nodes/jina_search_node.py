import logging
import asyncio
import aiohttp
from ...services.jina_api import get_jina_key
from ...framework.tool_registry import tool_registry
from ..base_tool_node import BaseToolNode 

logger = logging.getLogger(__name__)

@tool_registry(
    toolset_name="jina_search_and_visit",
    name="web_search",
    description="Searches the web for information about a question. This tool can execute search engine queries and return the titles, URLs, and snippets of the search results. Use this tool to find general information, recent developments, or different viewpoints on a topic. Specific and precise search queries should be used, avoiding overly broad terms. Each search should have a clear purpose, and queries should become more specific and targeted as research progresses.",
    parameters={
        "type": "object",
        "properties": {
            "purpose": {
                "type": "string",
                "description": "The purpose of the search, e.g., 'get general information', 'get recent developments', 'get different viewpoints'."
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of queries to search for, where each query is a string. Use precise, targeted queries, and consider using quotes to enclose phrases for more exact results. Do not visit more than 5 URLs.",
                "maxItems": 5
            }
        },
        "required": ["purpose", "queries"]
    },
    default_knowledge_item_type="SEARCH_RESULTS_LIST",
)
class JinaSearchNode(BaseToolNode):
    """
    [Refactored] Handles search queries using the Jina API.
    All logic is encapsulated within exec_async to conform to the BaseToolNode specification.
    """
    
    async def _fetch_single_query(self, query: str, purpose: str):
        """Private helper method to execute a single API call and return a standard result structure."""
        logger.debug("searching_query", extra={"query": query, "purpose": purpose})
        
        search_results_list = []
        success_flag = False
        error_message = None

        try:
            jina_key = get_jina_key()
            if not jina_key:
                error_message = "JINA_KEY environment variable is not set"
                logger.error("jina_api_key_missing", extra={"query": query})
            else:
                api_url = f'https://s.jina.ai/?q={query}'
                headers = {'Authorization': f'Bearer {jina_key}', 'X-Respond-With': 'no-content', 'Accept': 'application/json'}
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            data_json = await response.json()
                            search_results_list = data_json.get("data", [])
                            for res_item in search_results_list:
                                res_item.pop("usage", None)
                            success_flag = True
                        else:
                            error_text = await response.text()
                            error_message = f"Search engine returned an error: HTTP {response.status} - {error_text}"
                            # Log with appropriate severity based on error type
                            if response.status == 402:
                                logger.error("jina_api_insufficient_balance", extra={
                                    "query": query,
                                    "http_status": response.status,
                                    "error_detail": error_text,
                                    "action_required": "Recharge Jina API account at https://jina.ai"
                                })
                            elif response.status == 429:
                                logger.warning("jina_api_rate_limited", extra={
                                    "query": query,
                                    "http_status": response.status,
                                    "error_detail": error_text
                                })
                            else:
                                logger.error("jina_api_error", extra={
                                    "query": query,
                                    "http_status": response.status,
                                    "error_detail": error_text
                                })
        except asyncio.TimeoutError:
            error_message = "Search query timed out"
            logger.warning("jina_api_timeout", extra={"query": query})
        except Exception as e:
            error_message = f"Search error: {str(e)}"
            logger.error("jina_api_exception", extra={"query": query, "error": str(e)})

        # Prepare content for the LLM
        main_content_for_llm = {
            "query": query,
            "success": success_flag,
            "results_summary": f"Found {len(search_results_list)} results for '{query}'." if success_flag else f"[Search failed: {error_message}]",
            "results": search_results_list,
            "error_message": error_message if not success_flag else None
        }

        # Prepare content to be stored in the knowledge base
        knowledge_items_to_add = []
        if success_flag and search_results_list:
            knowledge_items_to_add.append({
                "item_type": "SEARCH_RESULTS_LIST",
                "content": search_results_list,
                "source_uri": f"jina_search_query://{query}",
                "metadata": {"query_string": query, "purpose": purpose}
            })
            
        return {"main_content_for_llm": main_content_for_llm, "_knowledge_items_to_add": knowledge_items_to_add}

    async def exec_async(self, prep_res: dict):
        """
        Core execution logic:
        1. Get parameters from prep_res.
        2. Check the Knowledge Base (KB) to separate cached queries from those needing a live fetch.
        3. Concurrently execute the queries that require a live fetch.
        4. Aggregate all results and construct the final return value.
        """
        tool_params = prep_res.get("tool_params", {})
        shared_context = prep_res.get("shared_context", {})
        
        queries = tool_params.get("queries", [])
        purpose = tool_params.get("purpose", "No purpose provided.")
        
        kb = shared_context.get('refs', {}).get('run', {}).get('runtime', {}).get("knowledge_base")
        queries_to_fetch = []
        cached_results_for_llm = []

        if kb:
            for query in queries:
                synthetic_uri = f"jina_search_query://{query}"
                existing_item = await kb.get_item_by_uri(synthetic_uri)
                if existing_item:
                    logger.info("search_query_found_in_kb", extra={"query": query})
                    results_from_kb = existing_item.get("content", [])
                    cached_results_for_llm.append({
                        "query": query, "success": True,
                        "results_summary": f"Found {len(results_from_kb)} results for '{query}' (from knowledge_base).",
                        "results": results_from_kb, "source": "knowledge_base"
                    })
                else:
                    queries_to_fetch.append(query)
        else:
            queries_to_fetch = queries

        live_fetch_results = []
        if queries_to_fetch:
            tasks = [self._fetch_single_query(q, purpose) for q in queries_to_fetch]
            live_fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_main_content_parts = list(cached_results_for_llm)
        all_knowledge_items_to_add = []

        for res in live_fetch_results:
            if isinstance(res, Exception):
                logger.error("parallel_jina_search_error", extra={"error": str(res)}, exc_info=res)
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
            "error_message": "One or more search queries failed." if is_error else None,
            "_knowledge_items_to_add": all_knowledge_items_to_add
        }

# Removed get_next_step_instructions

if __name__ == "__main__":
    # Standalone test for JinaSearchNode
    node = JinaSearchNode()
    
    # Create test state with the same structure as in main.py
    test_shared = {
        "state": {
            "question": "What is the PocketFlow framework?",
            "language": "zh",
            # "searched_queries": [], # Removed
            "visited_urls": [],
            "search_history": {}, # Changed to dict
            "visit_history": [],
            "messages": [],
            "current_action": {
                "type": "web_search",
                "query": "PocketFlow framework Python"
            },
            "current_tool_call_id": "call123"
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
        
        # Print results
        print(f"Node returned action: {action}")
        print(f"Number of items in search history: {len(test_shared['state']['search_history'])}")
        
        # Print the title of the first search result for a specific query, if it exists
        test_query = "PocketFlow framework Python"
        if test_query in test_shared["state"]["search_history"]:
            query_results = test_shared["state"]["search_history"][test_query]
            if query_results and len(query_results) > 0:
                first_result_item = query_results[0]
                print(f"Title of the first result for query '{test_query}': {first_result_item.get('title', 'No title')}")
            else:
                print(f"Query '{test_query}' returned no results.")
        else:
            print(f"Test query '{test_query}' not found in search history.")
    
    # Run the async test
    asyncio.run(run_test())
