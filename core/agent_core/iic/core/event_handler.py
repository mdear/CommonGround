import json
import os
from typing import Dict, List
import uuid
import asyncio
from pathlib import Path
from ...utils.serialization import get_serializable_run_snapshot
from .iic_handlers import get_iic_dir
from ..parser.parser import IICBlock
from . import core_logger as logger
from api.session import active_runs_store
from ...llm.call_llm import call_litellm_acompletion
from ...llm.config_resolver import LLMConfigResolver
from agent_profiles.loader import SHARED_LLM_CONFIGS
import re
from api.events import broadcast_project_structure_update

async def _save_run_state(run_context: dict, json_path: str):
    """Gets and saves a serializable snapshot of a run_context to the specified JSON file path."""
    try:
        snapshot_data = get_serializable_run_snapshot(run_context)
        # Use aiofiles for asynchronous writing
        import aiofiles
        async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(snapshot_data, ensure_ascii=False, indent=2))
        logger.debug("run_state_snapshot_saved", extra={"json_path": json_path})
    except Exception as e:
        logger.error("run_state_save_failed", extra={"json_path": json_path, "error": str(e)}, exc_info=True)

async def _save_minimal_iic_file(run_context: dict, iic_path: str):
    """Saves a minimal .iic file containing only metadata."""
    try:
        meta_info = run_context.get("meta", {})
        root_block = IICBlock(
            type="meta",
            attributes={
                "run_id": meta_info.get("run_id"),
                "project_id": run_context.get("project_id", "default"),
                "run_type": meta_info.get("run_type"),
                "created": meta_info.get("creation_timestamp"),
            },
            content=""
        )

        import aiofiles
        async with aiofiles.open(iic_path, 'w', encoding='utf-8') as f:
            await f.write(root_block.to_iic())
        logger.debug("minimal_iic_metadata_saved", extra={"iic_path": iic_path})
    except Exception as e:
        logger.error("minimal_iic_save_failed", extra={"iic_path": iic_path, "error": str(e)}, exc_info=True)


class EventHandler:
    """
    Event handler for managing events in the system.
    """

    def __init__(self):
        self.events: Dict[str, List[Dict]] = {} # run_id -> List of events
        self.query_block_ids: Dict[str, str] = {} # run_id -> query_block_id
        self.iic_files: Dict[str, str] = {}  # run_id -> iic file path
        self.run_locks: Dict[str, asyncio.Lock] = {} # run_id -> corresponding asyncio lock

    async def _trigger_intelligent_naming(self, run_id: str, initial_text: str):
        """
        Automatically generate a name for the IIC file based on the initial text of the run,
        or use a user-provided initial filename if available.
        """
        try:
            context = active_runs_store.get(run_id)
            if context:
                initial_filename = context.get("meta", {}).get("initial_filename")
                if initial_filename:
                    # Sanitize and slugify the user-provided name
                    slug = initial_filename.lower()
                    slug = re.sub(r'[^\w\s-]', '', slug).strip()
                    slug = re.sub(r'[\s_]+', '-', slug)
                    if not slug:
                        slug = f"run-summary-{run_id[:4]}"

                    logger.info("run_using_user_provided_filename", extra={"run_id": run_id, "initial_filename": initial_filename, "slug": slug})
                    await self.rename_iic_file(run_id, slug)
                    return # Important: exit after using the provided name

            if not initial_text:
                logger.info("run_no_initial_text_skipping_naming", extra={"run_id": run_id})
                return

            prompt = f"Please summarize the following user query into a concise, 4-5 word, filename-friendly English phrase (in slug format). Return only the filename, for example: 'research-ai-ethics-2024'. Query: '{initial_text}'"
            messages = [{"role": "user", "content": prompt}]

            # 1. Resolve fast_utils_llm config
            resolver = LLMConfigResolver(shared_llm_configs=SHARED_LLM_CONFIGS)
            fast_utils_llm_config = resolver.resolve({"llm_config_ref": "fast_utils_llm"})

            # 2. Call LLM
            response = await call_litellm_acompletion(
                messages=messages,
                llm_config=fast_utils_llm_config,
                stream=False
            )

            proposed_name = response.get("content", "").strip()

            # Clean and slugify the name
            slug = proposed_name.lower()
            slug = re.sub(r'[^\w\s-]', '', slug).strip()
            slug = re.sub(r'[\s_]+', '-', slug)
            if not slug:
                slug = f"run-summary-{run_id[:4]}"

            logger.info("llm_proposed_run_name", extra={"run_id": run_id, "proposed_name": proposed_name, "slug": slug})

            # Call the atomic rename method
            await self.rename_iic_file(run_id, slug)

        except Exception as e:
            logger.error("intelligent_naming_failed", extra={"run_id": run_id, "error": str(e)}, exc_info=True)

    async def rename_iic_file(self, run_id: str, new_name_slug: str):
        """Renames the .iic file and updates the project index. The .json filename remains the run_id."""
        lock = self.run_locks.get(run_id)
        if not lock:
            logger.error("cannot_rename_file_lock_not_found", extra={"run_id": run_id})
            return

        async with lock:
            from .iic_handlers import _read_project_index, _write_project_index
            old_path = self.iic_files.get(run_id)
            if not old_path:
                logger.error("cannot_rename_file_path_not_found", extra={"run_id": run_id})
                return

            project_dir = os.path.dirname(old_path)
            new_filename = f"{new_name_slug}.iic"
            new_path = os.path.join(project_dir, new_filename)

            # Handle filename conflicts
            if os.path.exists(new_path) and old_path != new_path:
                suffix = uuid.uuid4().hex[:4]
                new_filename_with_suffix = f"{new_name_slug}_{suffix}.iic"
                new_path = os.path.join(project_dir, new_filename_with_suffix)
                new_filename = new_filename_with_suffix  # for index update
                logger.warning("target_iic_filename_exists_renaming", extra={"target_filename": os.path.basename(new_path), "new_filename": new_filename_with_suffix})

            # 1. Update the index
            try:
                index_data = await _read_project_index(project_dir)
                if run_id in index_data or not os.path.exists(old_path): # Also update if it's a new file
                    index_data[run_id] = new_filename
                    await _write_project_index(project_dir, index_data)
                    logger.info("project_index_updated", extra={"run_id": run_id, "new_filename": new_filename})
            except Exception as e:
                logger.error("project_index_update_failed_rename", extra={"run_id": run_id, "error": str(e)}, exc_info=True)

            # 2. Update the authoritative path in memory
            self.iic_files[run_id] = new_path
            logger.info("authoritative_iic_path_updated", extra={"run_id": run_id, "new_path": new_path})

            # 3. If the old file actually exists at the old path, try to rename it
            if os.path.exists(old_path) and old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    logger.info("iic_file_renamed", extra={"old_filename": os.path.basename(old_path), "new_filename": os.path.basename(new_path)})
                except FileNotFoundError:
                    logger.warning("file_no_longer_exists_for_rename", extra={"old_path": old_path})
                except Exception as e:
                    logger.error("iic_file_rename_error", extra={"error": str(e)}, exc_info=True)

            # 4. Trigger a broadcast to notify the frontend to update the project structure.
            #    The new_name_slug here is the display name needed by the frontend (without .iic).
            try:
                asyncio.create_task(broadcast_project_structure_update(
                    "rename_run",
                    {"run_id": run_id, "new_name": new_name_slug}
                ))
                logger.info("project_structure_update_broadcast_triggered", extra={"run_id": run_id})
            except Exception as e:
                logger.error("broadcast_auto_rename_event_error", extra={"run_id": run_id, "error": str(e)}, exc_info=True)

    async def sync_run_to_iic(self, run_id):
        lock = self.run_locks.get(run_id)
        if not lock:
            logger.error("sync_run_to_iic_lock_not_found", extra={"run_id": run_id})
            return
        async with lock:
            from .iic_handlers import _read_project_index, _write_project_index
            authoritative_iic_path = self.iic_files.get(run_id)
            if not authoritative_iic_path:
                logger.error("sync_run_to_iic_file_path_not_found", extra={"run_id": run_id})
                return

            context = active_runs_store.get(run_id)
            if context is None:
                logger.error("context_not_found_for_run", extra={"run_id": run_id})
                return

            # --- Update project index ---
            try:
                project_id = context.get("project_id", "default")
                project_path = get_iic_dir(project_id)
                filename = os.path.basename(authoritative_iic_path)

                index_data = await _read_project_index(project_path)
                if run_id not in index_data or index_data[run_id] != filename:
                    index_data[run_id] = filename
                    await _write_project_index(project_path, index_data)
                    logger.info("run_added_to_project_index", extra={"run_id": run_id, "filename": filename, "project_path": project_path})
            except Exception as e:
                logger.error("project_index_update_failed_sync", extra={"run_id": run_id, "error": str(e)}, exc_info=True)

            # --- Persist run state and IIC file ---
            iic_path_obj = Path(authoritative_iic_path)
            json_path = str(iic_path_obj.parent / f"{run_id}.json")

            await _save_run_state(context, json_path)
            await _save_minimal_iic_file(context, authoritative_iic_path)

    async def _initialize_run_persistence_if_needed(self, run_id: str) -> bool:
        """
        Initializes the file path and lock for a given run_id if not already done.
        Also triggers intelligent naming on first initialization.
        Returns True if initialization is successful or was already done, False otherwise.
        """
        if run_id in self.iic_files:
            return True # Already initialized

        context = active_runs_store.get(run_id)
        if not context:
            logger.warning("persistence_init_failed_context_not_found", extra={"run_id": run_id})
            return False

        # Check if it is a resumed run
        resumed_path = context.get("meta", {}).get("source_iic_path")

        if resumed_path and os.path.exists(resumed_path):
            # This is a resumed run, use the existing path directly
            self.iic_files[run_id] = resumed_path
            self.run_locks[run_id] = asyncio.Lock()
            logger.info("persistence_initialized_for_resumed_run", extra={"run_id": run_id, "resumed_path": resumed_path})
            # **Crucial: Do not trigger _trigger_intelligent_naming**
        else:
            # This is a new run, execute the original logic
            project_id = context.get("project_id", "default")
            iic_dir = get_iic_dir(project_id)
            initial_path = os.path.join(iic_dir, f"{run_id}.iic")

            self.iic_files[run_id] = initial_path
            self.run_locks[run_id] = asyncio.Lock()
            logger.info("persistence_initialized_for_new_run", extra={"run_id": run_id, "initial_path": initial_path})

            # Only new runs trigger intelligent naming
            initial_text = context.get("team_state", {}).get("question", "")
            if initial_text:
                asyncio.create_task(self._trigger_intelligent_naming(run_id, initial_text))

        return True

    async def on_message(self, message_json):
        """
        Handle incoming messages.

        Args:
            message_json (dict): The message to handle.
        """
        try:
            body = json.loads(message_json)
            if not body:
                return

            msg_type = body.get("type", "")
            session_id = body.get("session_id", "")

            # Extract run_id based on message type, as 'run_ready' has it nested
            run_id = None
            if msg_type == "run_ready":
                run_id = body.get("data", {}).get("run_id")
            else:
                run_id = body.get("run_id")

            if not run_id:
                return

            logger.debug("callback_message_received", extra={"msg_type": msg_type, "run_id": run_id})

            # Initialize query_block_id if it's the first time we see this run_id
            if self.query_block_ids.get(run_id) is None:
                self.query_block_ids[run_id] = str(uuid.uuid4())

            # Persistence logic is now triggered by 'turn_completed'
            if msg_type == "turn_completed":
                logger.debug("persistence_triggered_by_event", extra={"msg_type": msg_type, "run_id": run_id})

                # Initialize persistence if it's the first time for this run
                if await self._initialize_run_persistence_if_needed(run_id):
                    # Call the core persistence function
                    try:
                        await self.sync_run_to_iic(run_id)

                        # Also save checkpoint for connection manager (for reconnection resilience)
                        try:
                            from api.connection_manager import connection_manager
                            context = active_runs_store.get(run_id)
                            if context:
                                # Save a lightweight checkpoint for quick reconnection
                                checkpoint_data = {
                                    "run_id": run_id,
                                    "status": context.get("meta", {}).get("status"),
                                    "turn_count": len(context.get("team_state", {}).get("turns", [])),
                                    "has_knowledge_base": bool(context.get("team_state", {}).get("knowledge_base"))
                                }
                                connection_manager.save_checkpoint(run_id, checkpoint_data)
                        except Exception as e:
                            logger.debug("checkpoint_save_skipped", extra={"run_id": run_id, "reason": str(e)})
                    except Exception as e:
                        logger.error("sync_run_to_iic_failed", extra={"run_id": run_id, "error": str(e)}, exc_info=True)
                else:
                    logger.error("skipping_persistence_initialization_failed", extra={"run_id": run_id})


        except json.JSONDecodeError:
            logger.error("message_json_decode_failed", extra={"raw_json": message_json})
        except Exception as e:
            logger.error("event_handler_message_failed", extra={"error": str(e)}, exc_info=True)

