#!/usr/bin/env python3
"""
Live Session Query Tool

Query running sessions in real-time via WebSocket API with pagination support.
Unlike analyze_session.py (which reads persisted JSON), this tool queries
the in-memory state of active sessions.

Usage:
    # Summary mode (default - always fast)
    python live_session_query.py <run_id> [--server URL]
    
    # Full context (may fail for large sessions)
    python live_session_query.py <run_id> --mode full
    
    # Specific section
    python live_session_query.py <run_id> --mode section --section team_state
    
    # Paginated messages from a specific context
    python live_session_query.py <run_id> --mode section --section sub_contexts \\
        --context _principal_context_ref --offset 0 --limit 20
    
    # Reconstruct full state by pulling all pages (for analysis)
    python live_session_query.py <run_id> --reconstruct --output snapshot.json

Examples:
    python live_session_query.py burrowing-cream-fulmar
    python live_session_query.py burrowing-cream-fulmar --mode section --section team_state
    python live_session_query.py burrowing-cream-fulmar --reconstruct --output live_snapshot.json

Note: Default port is read from core/.env (BACKEND_PORT setting)

Requirements:
    pip install websockets aiohttp
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


def get_port_from_env(var_name: str, default: int) -> int:
    """Read a port from core/.env file (single source of truth)."""
    script_dir = Path(__file__).parent
    env_file = script_dir.parent / "core" / ".env"
    
    if env_file.exists():
        try:
            content = env_file.read_text()
            match = re.search(rf'^{var_name}=(\d+)', content, re.MULTILINE)
            if match:
                return int(match.group(1))
        except Exception:
            pass
    
    return default


DEFAULT_BACKEND_PORT = get_port_from_env("BACKEND_PORT", 8800)


class LiveSessionClient:
    """WebSocket client for querying live session state with pagination support."""
    
    def __init__(self, server_url: str = None):
        if server_url is None:
            server_url = f"ws://127.0.0.1:{DEFAULT_BACKEND_PORT}"
        self.server_url = server_url
        self.http_url = server_url.replace("ws://", "http://").replace("wss://", "https://")
        self.session_id = None
        self.ws = None
    
    async def connect(self):
        """Establish connection to server."""
        import aiohttp
        import websockets
        
        print(f"Connecting to {self.http_url}...")
        
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(f"{self.http_url}/session") as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to get session token: {resp.status}")
                session_data = await resp.json()
                self.session_id = session_data.get("session_id")
                print(f"Got session token: {self.session_id[:20]}...")
        
        ws_url = f"{self.server_url}/ws/{self.session_id}"
        print(f"Connecting to WebSocket: {ws_url}")
        self.ws = await websockets.connect(ws_url, max_size=20 * 1024 * 1024)  # 20MB limit
        return self
    
    async def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()
    
    async def request_context(
        self,
        run_id: str,
        mode: str = "summary",
        section: Optional[str] = None,
        context_name: Optional[str] = None,
        work_module_id: Optional[str] = None,
        archive_index: Optional[int] = None,
        message_offset: int = 0,
        message_limit: int = 50
    ) -> Dict[str, Any]:
        """Send a request_run_context message and return the response."""
        request_data = {
            "run_id": run_id,
            "mode": mode
        }
        if section:
            request_data["section"] = section
        if context_name:
            request_data["context_name"] = context_name
        if work_module_id:
            request_data["work_module_id"] = work_module_id
        if archive_index is not None:
            request_data["archive_index"] = archive_index
        request_data["message_offset"] = message_offset
        request_data["message_limit"] = message_limit
        
        request = {
            "type": "request_run_context",
            "data": request_data
        }
        
        await self.ws.send(json.dumps(request))
        response_raw = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
        response = json.loads(response_raw)
        
        if response.get("type") != "run_context_response":
            raise Exception(f"Unexpected response type: {response.get('type')}")
        
        if "error" in response:
            raise Exception(response["error"])
        
        return response.get("data", {}).get("context", {})
    
    async def reconstruct_full_state(self, run_id: str, message_page_size: int = 100) -> Dict[str, Any]:
        """
        Reconstruct complete session state by pulling all sections with pagination.
        
        This method fetches data in small chunks to avoid WebSocket message size limits.
        For large sessions (many work modules with context_archive), it fetches each
        work module's archives separately with message pagination.
        
        Returns a structure compatible with the persisted JSON format for use with analyze_session.py.
        """
        print(f"\nReconstructing full state for: {run_id}")
        print("=" * 60)
        
        # Step 1: Get summary to understand the structure
        print("  [1/6] Fetching summary...")
        summary = await self.request_context(run_id, mode="summary")
        
        # Step 2: Get meta section
        print("  [2/6] Fetching metadata...")
        meta_response = await self.request_context(run_id, mode="section", section="meta")
        meta = meta_response.get("data", {})
        
        # Step 3: Get team_state WITHOUT context_archive (lightweight)
        print("  [3/6] Fetching team state (lightweight)...")
        team_response = await self.request_context(run_id, mode="section", section="team_state")
        team_state = team_response.get("data", {})
        work_module_summaries = team_response.get("work_module_summaries", {})
        
        # Step 4: Fetch context_archive for each work module that has archives
        print("  [4/6] Fetching work module archives...")
        work_modules = team_state.get("work_modules", {})
        
        for wm_id, wm_summary in work_module_summaries.items():
            archive_count = wm_summary.get("archive_count", 0)
            if archive_count == 0:
                continue
            
            print(f"       {wm_id}: {archive_count} archive(s)")
            
            # Initialize context_archive in the work module
            if wm_id in work_modules:
                work_modules[wm_id]["context_archive"] = []
            
            # Fetch each archive with message pagination
            for archive_summary in wm_summary.get("archives", []):
                arch_idx = archive_summary.get("archive_index", 0)
                total_messages = archive_summary.get("message_count", 0)
                
                # Fetch archive messages in pages
                all_messages = []
                offset = 0
                archive_data = {}
                
                while True:
                    arch_response = await self.request_context(
                        run_id,
                        mode="section",
                        section="team_state",
                        work_module_id=wm_id,
                        archive_index=arch_idx,
                        message_offset=offset,
                        message_limit=message_page_size
                    )
                    
                    data = arch_response.get("data", {})
                    messages = data.get("messages", [])
                    all_messages.extend(messages)
                    
                    # Capture non-message fields from first response
                    if not archive_data:
                        archive_data = {k: v for k, v in data.items() if k != "messages"}
                    
                    pagination = arch_response.get("pagination", {})
                    returned = pagination.get("returned", 0)
                    offset += returned
                    
                    if not pagination.get("has_more", False) or returned == 0:
                        break
                
                # Build complete archive
                archive_data["messages"] = all_messages
                work_modules[wm_id]["context_archive"].append(archive_data)
                
                if total_messages > message_page_size:
                    print(f"         archive[{arch_idx}]: {len(all_messages)}/{total_messages} messages")
        
        # Step 5: Get all sub_contexts with full message pagination
        print("  [5/6] Fetching agent contexts with messages...")
        sub_contexts_summary = summary.get("sub_contexts_summary", {})
        sub_contexts_state = {}
        
        for ctx_name, ctx_summary in sub_contexts_summary.items():
            total_messages = ctx_summary.get("message_count", 0)
            print(f"       {ctx_name}: {total_messages} messages")
            
            all_messages = []
            offset = 0
            data = {}
            
            while offset < total_messages:
                ctx_response = await self.request_context(
                    run_id,
                    mode="section",
                    section="sub_contexts",
                    context_name=ctx_name,
                    message_offset=offset,
                    message_limit=message_page_size
                )
                
                data = ctx_response.get("data", {})
                messages = data.get("messages", [])
                all_messages.extend(messages)
                
                pagination = ctx_response.get("pagination", {})
                returned = pagination.get("returned", 0)
                offset += returned
                
                if not pagination.get("has_more", False) or returned == 0:
                    break
            
            # Build the full context state
            sub_contexts_state[ctx_name] = {
                "messages": all_messages,
                "inbox": data.get("inbox", []),
                "deliverables": data.get("deliverables", {})
            }
        
        # Step 6: Get knowledge_base
        print("  [6/6] Fetching knowledge base...")
        try:
            kb_response = await self.request_context(run_id, mode="section", section="knowledge_base")
            knowledge_base = kb_response.get("data")
        except Exception:
            knowledge_base = None
        
        # Construct the full snapshot in persisted JSON format
        reconstructed = {
            "meta": meta,
            "team_state": team_state,
            "sub_contexts_state": sub_contexts_state,
            "knowledge_base": knowledge_base,
            "_reconstruction_metadata": {
                "source": "live_session_query",
                "reconstructed_at": datetime.now().isoformat(),
                "run_id": run_id,
                "server": self.server_url
            }
        }
        
        print(f"\nReconstruction complete!")
        print(f"  - Meta: {'present' if meta else 'empty'}")
        print(f"  - Team state: {len(work_modules)} work modules")
        total_archives = sum(len(wm.get("context_archive", [])) for wm in work_modules.values())
        print(f"  - Total archives: {total_archives}")
        print(f"  - Sub contexts: {len(sub_contexts_state)} contexts")
        total_msgs = sum(len(ctx.get("messages", [])) for ctx in sub_contexts_state.values())
        print(f"  - Total messages in sub_contexts: {total_msgs}")
        
        return reconstructed


async def reconstruct_and_save(
    run_id: str,
    server_url: str,
    output_path: str,
    page_size: int = 100
):
    """Reconstruct full session state and save to file."""
    try:
        import websockets
        import aiohttp
    except ImportError:
        print("Error: websockets and aiohttp packages required.")
        print("Install with: pip install websockets aiohttp")
        sys.exit(1)
    
    client = LiveSessionClient(server_url)
    try:
        await client.connect()
        reconstructed = await client.reconstruct_full_state(run_id, page_size)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(reconstructed, f, indent=2, default=str)
        
        print(f"\nSaved reconstructed state to: {output_path}")
        print(f"You can now analyze it with:")
        print(f"  python scripts/analyze_session.py {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await client.close()


async def query_live_session(
    run_id: str, 
    server_url: str = "ws://127.0.0.1:8000",
    mode: str = "summary",
    section: str = None,
    context_name: str = None,
    message_offset: int = 0,
    message_limit: int = 50,
    output_file: str = None
):
    """Query a live session's state via WebSocket with pagination support."""
    try:
        import websockets
    except ImportError:
        print("Error: websockets package required. Install with: pip install websockets")
        sys.exit(1)

    # Step 1: Get a session token via HTTP
    import aiohttp
    http_url = server_url.replace("ws://", "http://").replace("wss://", "https://")
    
    print(f"Connecting to {http_url}...")
    
    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(f"{http_url}/session") as resp:
                if resp.status != 200:
                    print(f"Error: Failed to get session token: {resp.status}")
                    sys.exit(1)
                session_data = await resp.json()
                session_id = session_data.get("session_id")
                print(f"Got session token: {session_id[:20]}...")
    except aiohttp.ClientError as e:
        print(f"Error connecting to server: {e}")
        print("Is the CommonGround server running?")
        sys.exit(1)

    # Step 2: Connect via WebSocket
    ws_url = f"{server_url}/ws/{session_id}"
    print(f"Connecting to WebSocket: {ws_url}")
    
    try:
        # Increase max message size to handle larger responses (10MB)
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
            # Build request with pagination options
            request_data = {
                "run_id": run_id,
                "mode": mode
            }
            if section:
                request_data["section"] = section
            if context_name:
                request_data["context_name"] = context_name
            request_data["message_offset"] = message_offset
            request_data["message_limit"] = message_limit
            
            request = {
                "type": "request_run_context",
                "data": request_data
            }
            
            print(f"Requesting context for run_id: {run_id} (mode={mode})")
            if section:
                print(f"  Section: {section}")
            if context_name:
                print(f"  Context: {context_name}, offset={message_offset}, limit={message_limit}")
            
            await ws.send(json.dumps(request))
            
            # Wait for response
            response_raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            response = json.loads(response_raw)
            
            if response.get("type") == "run_context_response":
                if "error" in response:
                    print(f"\nError: {response['error']}")
                    print("\nPossible reasons:")
                    print("  - Run ID not found (session may have ended)")
                    print("  - Session was never started with this ID")
                    print("  - Server restarted (in-memory state lost)")
                    sys.exit(1)
                
                context = response.get("data", {}).get("context", {})
                
                # Output to file if specified, otherwise print
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(context, f, indent=2, default=str)
                    print(f"\nSaved query results to: {output_file}")
                else:
                    print_live_context(context, run_id, mode)
            else:
                print(f"Unexpected response type: {response.get('type')}")
                print(json.dumps(response, indent=2)[:1000])
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
        if "message too big" in str(e).lower():
            print("\nThe response was too large. Try using:")
            print("  --mode summary   (default, always small)")
            print("  --mode section --section <name>   (specific section)")
    except asyncio.TimeoutError:
        print("Timeout waiting for response")
    except Exception as e:
        print(f"Error: {e}")


def print_live_context(context: dict, run_id: str, mode: str):
    """Pretty print the live session context based on mode."""
    print("\n" + "=" * 80)
    print(f"LIVE SESSION: {run_id}")
    print("=" * 80)
    print("(Queried from in-memory state - this is REAL-TIME data)")
    
    # Check for error
    if "error" in context:
        print(f"\nError: {context['error']}")
        return
    
    response_mode = context.get("mode", "unknown")
    print(f"Response mode: {response_mode}")
    
    if response_mode == "summary":
        print_summary_context(context)
    elif response_mode == "section":
        print_section_context(context)
    elif response_mode == "full":
        print_full_context(context, run_id)
    else:
        # Legacy format or unknown
        print_full_context(context, run_id)
    
    print("\n" + "=" * 80)


def print_summary_context(context: dict):
    """Print summary mode response."""
    # Meta
    meta = context.get("meta", {})
    print(f"\nStatus: {meta.get('status', 'unknown')}")
    print(f"Run Type: {meta.get('run_type', 'unknown')}")
    
    # Team State
    team_state = context.get("team_state", {})
    work_modules = team_state.get("work_modules", {})
    dispatch_history = team_state.get("dispatch_history", [])
    is_principal_running = team_state.get("is_principal_flow_running", False)
    
    print(f"\nPrincipal Running: {'YES' if is_principal_running else 'NO'}")
    
    # Work Modules
    print(f"\n--- Work Modules ({len(work_modules)}) ---")
    for wm_id, wm in sorted(work_modules.items()):
        status = wm.get("status", "unknown")
        title = wm.get("title", wm.get("name", "unnamed"))[:50]
        print(f"  {wm_id}: {status} - {title}")
    
    # Dispatch Summary
    running_dispatches = [d for d in dispatch_history if d.get("status") == "RUNNING"]
    completed_dispatches = [d for d in dispatch_history if "SUCCESS" in d.get("status", "")]
    
    print(f"\n--- Dispatches ---")
    print(f"  Total: {len(dispatch_history)}")
    print(f"  Running: {len(running_dispatches)}")
    print(f"  Completed: {len(completed_dispatches)}")
    
    if running_dispatches:
        print(f"\n  Currently Running:")
        for d in running_dispatches:
            module_id = d.get("module_id", "?")
            profile = d.get("profile_logical_name", "?")
            start = d.get("start_timestamp", "?")[:19] if d.get("start_timestamp") else "?"
            print(f"    {module_id} ({profile}) - started {start}")
    
    # Sub Contexts Summary
    sub_summaries = context.get("sub_contexts_summary", {})
    print(f"\n--- Agent Contexts (Summary) ---")
    for ctx_name, summary in sub_summaries.items():
        display_name = ctx_name.replace("_context_ref", "").replace("_", " ").title()
        msg_count = summary.get("message_count", 0)
        inbox_count = summary.get("inbox_count", 0)
        has_deliverables = summary.get("has_deliverables", False)
        
        print(f"\n  {display_name}:")
        print(f"    Messages: {msg_count}")
        print(f"    Inbox items: {inbox_count}")
        if has_deliverables:
            keys = summary.get("deliverable_keys", [])
            print(f"    Deliverables: {', '.join(keys) if keys else 'Yes'}")
        
        # Last message preview
        last_msg = summary.get("last_message")
        if last_msg:
            role = last_msg.get("role", "?")
            preview = last_msg.get("content_preview", "")[:80]
            print(f"    Last ({role}): {preview}...")
    
    # Knowledge Base Summary
    kb_summary = context.get("knowledge_base_summary", {})
    if kb_summary:
        print(f"\n--- Knowledge Base ({len(kb_summary)} entries) ---")
        for key, info in kb_summary.items():
            kb_type = info.get("type", "?")
            size = info.get("size", "?")
            print(f"    {key}: {kb_type} (size={size})")


def print_section_context(context: dict):
    """Print section mode response."""
    section = context.get("section", "unknown")
    print(f"\nSection: {section}")
    
    # Handle sub_contexts section specially (has pagination)
    if section == "sub_contexts":
        if "available_contexts" in context and "context_summaries" in context:
            # List of contexts
            print(f"\nAvailable contexts:")
            for ctx_name, summary in context.get("context_summaries", {}).items():
                print(f"  {ctx_name}: {summary.get('message_count', 0)} messages")
            print(f"\n{context.get('hint', '')}")
        elif "context_name" in context:
            # Specific context with pagination
            ctx_name = context.get("context_name")
            data = context.get("data", {})
            pagination = context.get("pagination", {})
            
            print(f"\nContext: {ctx_name}")
            print(f"Messages: {pagination.get('returned', 0)} of {pagination.get('total_messages', 0)}")
            print(f"Offset: {pagination.get('offset', 0)}, Limit: {pagination.get('limit', 50)}")
            print(f"Has more: {pagination.get('has_more', False)}")
            
            messages = data.get("messages", [])
            print(f"\n--- Messages ---")
            for i, msg in enumerate(messages):
                role = msg.get("role", "?")
                content = str(msg.get("content", ""))[:100]
                print(f"  [{pagination.get('offset', 0) + i}] {role}: {content}...")
            
            inbox = data.get("inbox", [])
            if inbox:
                print(f"\n--- Inbox ({len(inbox)} items) ---")
                for item in inbox[:5]:
                    print(f"    {str(item)[:80]}...")
            
            deliverables = data.get("deliverables", {})
            if deliverables:
                print(f"\n--- Deliverables ---")
                for key in deliverables.keys():
                    print(f"    {key}")
    else:
        # Other sections - just dump the data
        data = context.get("data")
        if data:
            if isinstance(data, dict):
                print(f"\n{json.dumps(data, indent=2, default=str)[:3000]}")
                if len(json.dumps(data, default=str)) > 3000:
                    print("... (truncated)")
            else:
                print(f"\n{str(data)[:3000]}")


def print_full_context(context: dict, run_id: str):
    """Print full context (legacy format)."""
    # Meta
    meta = context.get("meta", {})
    print(f"\nStatus: {meta.get('status', 'unknown')}")
    print(f"Run Type: {meta.get('run_type', 'unknown')}")
    
    # Team State
    team_state = context.get("team_state", {})
    work_modules = team_state.get("work_modules", {})
    dispatch_history = team_state.get("dispatch_history", [])
    is_principal_running = team_state.get("is_principal_flow_running", False)
    
    print(f"\nPrincipal Running: {'YES' if is_principal_running else 'NO'}")
    
    # Work Modules
    print(f"\n--- Work Modules ({len(work_modules)}) ---")
    for wm_id, wm in sorted(work_modules.items()):
        status = wm.get("status", "unknown")
        title = wm.get("title", wm.get("name", "unnamed"))[:50]
        print(f"  {wm_id}: {status} - {title}")
    
    # Dispatch History
    running_dispatches = [d for d in dispatch_history if d.get("status") == "RUNNING"]
    completed_dispatches = [d for d in dispatch_history if "SUCCESS" in d.get("status", "")]
    
    print(f"\n--- Dispatches ---")
    print(f"  Total: {len(dispatch_history)}")
    print(f"  Running: {len(running_dispatches)}")
    print(f"  Completed: {len(completed_dispatches)}")
    
    if running_dispatches:
        print(f"\n  Currently Running:")
        for d in running_dispatches:
            module_id = d.get("module_id", "?")
            profile = d.get("profile_logical_name", "?")
            start = d.get("start_timestamp", "?")[:19] if d.get("start_timestamp") else "?"
            print(f"    {module_id} ({profile}) - started {start}")
    
    # Sub Contexts (live agent state)
    sub_contexts = context.get("sub_contexts_state", {})
    print(f"\n--- Agent Contexts (Live) ---")
    
    for ctx_name, ctx_state in sub_contexts.items():
        if not isinstance(ctx_state, dict):
            continue
        messages = ctx_state.get("messages", [])
        inbox = ctx_state.get("inbox", [])
        deliverables = ctx_state.get("deliverables", {})
        
        # Clean up name
        display_name = ctx_name.replace("_context_ref", "").replace("_", " ").title()
        print(f"\n  {display_name}:")
        print(f"    Messages: {len(messages)}")
        print(f"    Inbox items: {len(inbox)}")
        if deliverables:
            print(f"    Has deliverables: Yes")
        
        # Show last message preview
        if messages:
            last_msg = messages[-1]
            role = last_msg.get("role", "?")
            content = str(last_msg.get("content", ""))[:100]
            print(f"    Last message ({role}): {content}...")


def main():
    parser = argparse.ArgumentParser(
        description="Query live session state via WebSocket API with pagination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  summary      Lightweight overview without full messages (default, always fast)
  full         Complete snapshot (may fail for large sessions)
  section      Specific section with optional pagination
  
Special Options:
  --reconstruct  Pull all sections with pagination and reconstruct full state
                 Output is compatible with analyze_session.py

Sections (for --mode section):
  meta          Session metadata
  team_state    Work modules and dispatch history
  sub_contexts  Agent contexts with message pagination
  knowledge_base Knowledge base entries

Examples:
    # Quick summary (default)
    python live_session_query.py burrowing-cream-fulmar
    
    # Get team state only
    python live_session_query.py <run_id> --mode section --section team_state
    
    # List available contexts
    python live_session_query.py <run_id> --mode section --section sub_contexts
    
    # Get paginated messages from principal context
    python live_session_query.py <run_id> --mode section --section sub_contexts \\
        --context _principal_context_ref --offset 0 --limit 20
    
    # Reconstruct full state and save to file (for use with analyze_session.py)
    python live_session_query.py <run_id> --reconstruct --output live_snapshot.json
    
    # Then analyze with:
    python scripts/analyze_session.py live_snapshot.json
    
Note: This queries IN-MEMORY state. If the server restarted, the session
won't be found even if it was persisted to JSON.
        """
    )
    parser.add_argument("run_id", help="The run ID to query")
    parser.add_argument("--server", default=f"ws://127.0.0.1:{DEFAULT_BACKEND_PORT}", 
                       help=f"WebSocket server URL (default: ws://127.0.0.1:{DEFAULT_BACKEND_PORT})")
    parser.add_argument("--mode", choices=["summary", "full", "section"], default="summary",
                       help="Query mode: summary (default), full, or section")
    parser.add_argument("--section", choices=["meta", "team_state", "sub_contexts", "knowledge_base"],
                       help="Section to retrieve (for --mode section)")
    parser.add_argument("--context", dest="context_name",
                       help="Context name for sub_contexts section (e.g., _principal_context_ref)")
    parser.add_argument("--offset", type=int, default=0,
                       help="Message offset for pagination (default: 0)")
    parser.add_argument("--limit", type=int, default=50,
                       help="Message limit for pagination (default: 50)")
    
    # Reconstruct options
    parser.add_argument("--reconstruct", action="store_true",
                       help="Reconstruct full state by pulling all pages")
    parser.add_argument("--output", "-o", 
                       help="Output file for query results (JSON). Works with all modes.")
    parser.add_argument("--page-size", type=int, default=100,
                       help="Page size for message pagination during reconstruction (default: 100)")
    
    args = parser.parse_args()
    
    # Handle reconstruct mode
    if args.reconstruct:
        output_path = args.output or f"{args.run_id}_live.json"
        asyncio.run(reconstruct_and_save(
            args.run_id,
            args.server,
            output_path,
            args.page_size
        ))
        return
    
    # Validate arguments for regular query
    if args.mode == "section" and not args.section:
        parser.error("--section is required when --mode is 'section'")
    
    asyncio.run(query_live_session(
        args.run_id, 
        args.server,
        mode=args.mode,
        section=args.section,
        context_name=args.context_name,
        message_offset=args.offset,
        message_limit=args.limit,
        output_file=args.output
    ))


if __name__ == "__main__":
    main()
