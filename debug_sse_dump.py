"""Dump all raw SSE events from Mira to understand sub-agent event flow."""
import asyncio
import json
import os
import sys

sys.path.insert(0, "/opt/tiger/mira_nas/userdata/7096375/deer-flow")
from mira_client import MiraClient

TOKEN = os.environ.get("MIRA_SESSION", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ2IjoidjEiLCJ0biI6Im1pcmEiLCJ1aWQiOiI3MDk2Mzc1Iiwic2MiOiJ1c2VyIiwiaXNzIjoibWlyYSIsImV4cCI6MTc3NjQ4MTAxNCwiaWF0IjoxNzczODg5MDE0fQ.knM-5DTILLOncucGycIlHKoIkfoXhPXtfvI3hjvJkYA")

async def main():
    async with MiraClient(TOKEN) as client:
        sid = await client.create_session()
        print(f"Session: {sid}\n")
        
        count = 0
        async for evt in client.chat(sid, "1+1等于几", mode="quick"):
            count += 1
            print(f"--- Event #{count} ---")
            print(f"  event_type: {evt.event}")
            print(f"  inner_type: {evt.inner_type}")
            print(f"  block_type: {evt.block_type}")
            print(f"  delta_type: {evt.delta_type}")
            print(f"  text: {repr(evt.text[:200]) if evt.text else '(empty)'}")
            print(f"  message_id: {evt.message_id}")
            print(f"  session_id: {evt.session_id}")
            # Print raw data keys
            if isinstance(evt.data, dict):
                print(f"  data_keys: {list(evt.data.keys())}")
                # Check for special fields that might indicate sub-agent/tool usage
                for key in ['status', 'type', 'event', 'tool', 'tool_name', 'tool_use', 'action', 'stage', 'step']:
                    if key in evt.data:
                        val = evt.data[key]
                        if isinstance(val, (str, int, bool)):
                            print(f"  data.{key}: {val}")
                        elif isinstance(val, dict):
                            print(f"  data.{key}: {json.dumps(val, ensure_ascii=False)[:200]}")
            print()
        
        print(f"\nTotal events: {count}")
        
        # Clean up
        await client.delete_session(sid)

asyncio.run(main())
