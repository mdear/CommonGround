import logging
import asyncio
import anyio
from typing import Dict, Any

# Import the new base class
from .base_tool_node import BaseToolNode
from ..services.server_manager import reconnect_mcp_server

logger = logging.getLogger(__name__)

# Maximum number of reconnection attempts before giving up
MAX_RECONNECT_ATTEMPTS = 2


class MCPProxyNode(BaseToolNode):
    """
    [Refactored] A proxy node for transparently calling tools on a native MCP server.
    Now inherits from BaseToolNode, which encapsulates standard prep and post logic.
    Supports automatic reconnection on connection failures.
    """
    def __init__(self, unique_tool_name: str, original_tool_name: str, server_name: str, tool_info: Dict, **kwargs):
        # 1. [Core Fix] Set the _tool_info attribute before calling the parent class constructor
        self._tool_info = tool_info

        # 2. Now it's safe to call the parent's constructor, which will perform checks
        # Note: We no longer need max_retries and wait, as BaseToolNode handles them
        # But for safety, we pass them via kwargs
        super().__init__(**kwargs)

        # 3. Continue with this class's initialization logic
        self.unique_tool_name = unique_tool_name
        self.original_tool_name = original_tool_name
        self.server_name = server_name
        logger.debug("mcp_proxy_node_initialized", extra={"unique_tool_name": self.unique_tool_name, "server_name": self.server_name})

    # The prep_async method has been removed; its logic is now in exec_async

    async def _call_tool_with_reconnect(self, session_group, tool_params: Dict, attempt: int = 1) -> Dict[str, Any]:
        """
        Internal method that attempts to call the MCP tool, with automatic reconnection on failure.

        Args:
            session_group: The MCP session group to use
            tool_params: Parameters to pass to the tool
            attempt: Current attempt number (1-indexed)

        Returns:
            Result dictionary with status and payload/error
        """
        try:
            # Call the MCP tool with a 60-second timeout
            result = await asyncio.wait_for(
                session_group.call_tool(self.original_tool_name, tool_params),
                timeout=60.0
            )

            if result is None or not hasattr(result, 'content'):
                raise ValueError("MCP tool returned an invalid or null response.")

            logger.info("mcp_proxy_tool_call_success", extra={
                "unique_tool_name": self.unique_tool_name,
                "original_tool_name": self.original_tool_name,
                "attempt": attempt
            })

            # Extract content
            content_text = ""
            if result.content and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, 'text') and content_item.text is not None:
                    content_text = content_item.text
                else:
                    content_text = str(content_item)

            return {
                "status": "success",
                "payload": {
                    "status": "success",
                    "tool_name_invoked": self.unique_tool_name,
                    "server_name": self.server_name,
                    "response_preview": content_text
                }
            }

        except anyio.ClosedResourceError as e:
            # Connection was closed - attempt reconnection if we haven't exceeded max attempts
            if attempt < MAX_RECONNECT_ATTEMPTS:
                logger.warning("mcp_proxy_connection_closed_attempting_reconnect", extra={
                    "unique_tool_name": self.unique_tool_name,
                    "server_name": self.server_name,
                    "attempt": attempt,
                    "max_attempts": MAX_RECONNECT_ATTEMPTS
                })

                # Attempt to reconnect the server
                reconnect_success = await reconnect_mcp_server(session_group, self.server_name)

                if reconnect_success:
                    logger.info("mcp_proxy_reconnect_success_retrying", extra={
                        "unique_tool_name": self.unique_tool_name,
                        "server_name": self.server_name,
                        "attempt": attempt + 1
                    })
                    # Retry the tool call with the reconnected session
                    return await self._call_tool_with_reconnect(session_group, tool_params, attempt + 1)
                else:
                    logger.error("mcp_proxy_reconnect_failed", extra={
                        "unique_tool_name": self.unique_tool_name,
                        "server_name": self.server_name
                    })

            # If we're here, either max attempts exceeded or reconnection failed
            error_msg = f"The connection to the external service '{self.server_name}' required by the tool '{self.unique_tool_name}' was unexpectedly closed. Reconnection attempt{'s' if attempt > 1 else ''} failed."
            logger.error("mcp_proxy_connection_closed_final", extra={
                "unique_tool_name": self.unique_tool_name,
                "attempts_made": attempt
            }, exc_info=True)

            return {
                "status": "error",
                "error_message": error_msg,
                "payload": {
                    "type": "CRITICAL_CONNECTION_FAILURE",
                    "summary": error_msg,
                    "instruction_for_llm": (
                        f"**Action Required: You MUST NOT try to call the '{self.unique_tool_name}' tool or any other tool from the '{self.server_name}' toolset again.**\n"
                        "1. Acknowledge this critical failure in your reasoning.\n"
                        "2. Call the 'finish_flow' tool with a reason explaining that the task could not be completed due to a persistent external service connection error."
                    )
                }
            }

        except asyncio.TimeoutError:
            error_msg = f"Call to native MCP tool '{self.original_tool_name}' (server: {self.server_name}) timed out."
            logger.error("mcp_proxy_tool_timeout", extra={
                "unique_tool_name": self.unique_tool_name,
                "original_tool_name": self.original_tool_name,
                "server_name": self.server_name,
                "attempt": attempt
            })
            return {"status": "error", "error_message": error_msg}

        except Exception as e:
            error_msg = f"Error calling native MCP tool '{self.original_tool_name}' (server: {self.server_name}): {str(e)}"
            logger.error("mcp_proxy_tool_call_error", extra={
                "unique_tool_name": self.unique_tool_name,
                "original_tool_name": self.original_tool_name,
                "server_name": self.server_name,
                "attempt": attempt
            }, exc_info=True)
            return {"status": "error", "error_message": error_msg}

    async def exec_async(self, prep_res: Dict) -> Dict[str, Any]:
        """
        [Refactored] This is the core execution logic for this tool node.
        It follows the BaseToolNode contract, receiving prep_res and returning a standard result dictionary.
        Now includes automatic reconnection on connection failures.
        """
        # 1. Get standard input from prep_res
        tool_params = prep_res.get("tool_params", {})
        shared_context = prep_res.get("shared_context", {})

        # 2. Get specific resources needed by this node from shared_context
        session_group = shared_context.get("runtime_objects", {}).get("mcp_session_group")
        if not session_group:
            error_msg = f"MCPProxyNode ({self.unique_tool_name}): Agent's context-specific MCP Session Group not found."
            logger.error("mcp_proxy_session_group_not_found", extra={"unique_tool_name": self.unique_tool_name})
            return {"status": "error", "error_message": error_msg}

        # 3. Execute the tool call with automatic reconnection support
        logger.info("mcp_proxy_tool_call_begin", extra={"unique_tool_name": self.unique_tool_name})

        return await self._call_tool_with_reconnect(session_group, tool_params)

    # The post_async method has been removed because BaseToolNode handles it
