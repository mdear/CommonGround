import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMConfigResolver:
    """
    A centralized, authoritative LLM configuration resolver.
    It is responsible for converting self-describing YAML configuration objects
    into the final LiteLLM parameter dictionary.
    """
    def __init__(self, shared_llm_configs: Dict):
        """
        Initializes the resolver.

        Args:
            shared_llm_configs (Dict): The LLM configuration store loaded from the loader, with inheritance already resolved.
        """
        self.shared_llm_configs = shared_llm_configs

    def _recursive_resolve(self, config_value: Any) -> Any:
        """
        Recursively processes configuration values, parsing _type directives.
        """
        if not isinstance(config_value, dict) or "_type" not in config_value:
            return config_value

        directive = config_value["_type"]

        if directive == "from_env":
            var_name = config_value.get("var")
            if not var_name:
                raise ValueError(f"'_type: from_env' directive is missing the 'var' key in config: {config_value}")

            env_value = os.getenv(var_name)
            if env_value is not None:
                # Try to convert string "true" / "false" to boolean, "null" to None
                if env_value.lower() == 'true': return True
                if env_value.lower() == 'false': return False
                if env_value.lower() == 'null': return None

                # Try to parse as JSON (for extra_headers and other complex types)
                if env_value.strip().startswith('{') or env_value.strip().startswith('['):
                    try:
                        return json.loads(env_value)
                    except json.JSONDecodeError:
                        logger.warning("env_var_json_parse_failed", extra={
                            "var_name": var_name,
                            "hint": "Value looks like JSON but failed to parse. Using as string."
                        })

                # Try to convert numeric strings to int/float
                try:
                    # Try int first
                    if '.' not in env_value and 'e' not in env_value.lower():
                        return int(env_value)
                    # Then try float
                    return float(env_value)
                except ValueError:
                    pass  # Not a number, return as string

                return env_value

            if "default" in config_value:
                return config_value["default"]

            if config_value.get("required", False):
                raise ValueError(f"Required environment variable '{var_name}' is not set and no default was provided.")

            return None

        if directive == "json_from_file":
            path_str = config_value.get("path")
            if not path_str:
                raise ValueError(f"'_type: json_from_file' directive is missing the 'path' key in config: {config_value}")

            if not os.path.exists(path_str):
                 raise FileNotFoundError(f"File specified in 'json_from_file' not found: {path_str}")

            with open(path_str, 'r', encoding='utf-8') as f:
                return json.load(f)

        logger.warning("unknown_type_directive", extra={"directive": directive})
        return config_value

    def resolve(self, agent_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        The main resolution function. Generates the final LiteLLM parameters based on the Agent Profile.
        """
        from agent_profiles.loader import get_active_llm_config_by_name # Delayed import to avoid circular dependency

        llm_config_ref = agent_profile.get("llm_config_ref")
        if not llm_config_ref:
            raise ValueError(f"Profile '{agent_profile.get('name')}' is missing 'llm_config_ref'.")

        base_config = get_active_llm_config_by_name(self.shared_llm_configs, llm_config_ref)
        if not base_config:
            raise ValueError(f"LLM Config '{llm_config_ref}' not found or is inactive.")

        raw_config = base_config.get("config", {}).copy()

        final_params = {}
        for key, value in raw_config.items():
            try:
                resolved_value = self._recursive_resolve(value)
                if resolved_value is not None:
                    final_params[key] = resolved_value
            except (ValueError, FileNotFoundError) as e:
                logger.error("config_key_resolution_error", extra={"key": key, "llm_config_ref": llm_config_ref, "error_message": str(e)})
                # Depending on requirements, one can choose to throw an exception here or continue, leaving the configuration incomplete
                raise e

        if "litellm_options" in final_params:
            options = final_params.pop("litellm_options")
            if isinstance(options, dict):
                final_params.update(options)

        return final_params
