# utils/event_strategies.py

import logging
from typing import Dict, Callable, Any

logger = logging.getLogger(__name__)

class EventHandlingStrategy:
    """Defines how to handle a specific source of InboxItem."""
    def __init__(self, ingestor_func: Callable, default_injection_mode: str, default_params: Dict[str, Any]):
        self.ingestor = ingestor_func
        self.default_injection_mode = default_injection_mode
        self.default_params = default_params

# Import all Ingestor functions to be created
from .ingestors import (
    tool_result_ingestor,
    generic_message_ingestor,
    markdown_formatter_ingestor,
    templated_content_ingestor,
    work_modules_ingestor,
    principal_history_summary_ingestor,
    json_history_ingestor,
    tagged_content_ingestor,
    observer_failure_ingestor,
    user_prompt_ingestor,
    protocol_aware_ingestor,
    principal_completed_ingestor
)

EVENT_STRATEGY_REGISTRY: Dict[str, EventHandlingStrategy] = {
    "TOOL_RESULT": EventHandlingStrategy(
        ingestor_func=tool_result_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "tool", "is_persistent_in_memory": True}
    ),
    "AGENT_STARTUP_BRIEFING": EventHandlingStrategy(
        ingestor_func=protocol_aware_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
    "SELF_REFLECTION_PROMPT": EventHandlingStrategy(
        ingestor_func=templated_content_ingestor,
        default_injection_mode="prepend_to_role",
        default_params={"role": "user"}
    ),
    "INTERNAL_DIRECTIVE": EventHandlingStrategy(
        ingestor_func=templated_content_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
    "PARTNER_DIRECTIVE": EventHandlingStrategy(
        ingestor_func=markdown_formatter_ingestor, # <-- Changed to a more powerful formatting tool
        default_injection_mode="append_as_new_message",
        default_params={
            "role": "user", 
            "is_persistent_in_memory": True,
            "title": "### Directive from Partner", # <-- Add a title to make the context clearer for the LLM
            "key_renames": { "content": "Instruction" } # <-- Rename the 'content' key in the payload to 'Instruction'
        }
    ),
    "PRINCIPAL_COMPLETED": EventHandlingStrategy(
        ingestor_func=principal_completed_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
    "ALL_WORK_COMPLETED_PROMPT": EventHandlingStrategy(
        ingestor_func=generic_message_ingestor,
        default_injection_mode="prepend_to_role",
        default_params={"role": "user"}
    ),
    "PROFILES_UPDATED_NOTIFICATION": EventHandlingStrategy(
        ingestor_func=generic_message_ingestor,
        default_injection_mode="prepend_to_role",
        default_params={"role": "user"}
    ),
    "WORK_MODULES_STATUS_UPDATE": EventHandlingStrategy(
        ingestor_func=work_modules_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": False}
    ),
    "PRINCIPAL_ACTIVITY_UPDATE": EventHandlingStrategy(
        ingestor_func=principal_history_summary_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": False} # Transient information, not saved to history
    ),
    "FIM_INSTRUCTION": EventHandlingStrategy(
        ingestor_func=templated_content_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
    "JSON_HISTORY_FOR_LLM": EventHandlingStrategy(
        ingestor_func=json_history_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
    "TOOL_INPUTS_BRIEFING": EventHandlingStrategy(
        ingestor_func=markdown_formatter_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
    "ORIGINAL_QUESTION": EventHandlingStrategy(
        ingestor_func=tagged_content_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
    "OBSERVER_FAILURE": EventHandlingStrategy(
        ingestor_func=observer_failure_ingestor,
        default_injection_mode="append_as_new_message", # Append as a separate message
        default_params={"role": "system", "is_persistent_in_memory": False} # Transient system-level error
    ),
    "USER_PROMPT": EventHandlingStrategy(
        ingestor_func=user_prompt_ingestor,
        default_injection_mode="append_as_new_message",
        default_params={"role": "user", "is_persistent_in_memory": True}
    ),
}

logger.info("event_strategy_registry_initialized", extra={"count": len(EVENT_STRATEGY_REGISTRY)})
