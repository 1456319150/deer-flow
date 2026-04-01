"""Dump all raw SSE events from a tool-triggering query."""
import asyncio
import json
import os
import sys

sys.path.insert(0, "/opt/tiger/mira_nas/userdata/7096375/deer-flow")
from mira_client import MiraClient

TOKEN = os.environ.get("MIRA_SESSION", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ2IjoidjEiLCJ0biI6Im1pcmEiLCJ1aWQiOiI3MDk2Mzc1Iiwic2MiOiJ1c2VyIiwiaXNzIjoibWlyYSIsImV4cCI6MTc3NjQ4MTAxNCwiaWF0IjoxNzczODg5MDE0fQ.knM-5DTILLOncucGycIlHKoIkfoXhPXtfvI3hjvJkYA")

OUTPUT_FILE = "/opt/tiger/mira_nas/userdata/7096375/deer-flow/debug_sse_output.txt"

async def main():
    lines = []
    def log(s=""):
        lines.append(s)
        print(s, flush=True)
    
    async with MiraClient(TOKEN) as client:
        sid = await client.create_session()
        log(f"Session: {sid}\n")
        
        count = 0
        try:
            async for evt in client.chat(sid, "今天北京天气怎么样", mode="quick"):
                count += 1
                log(f"--- Event #{count} ---")
                log(f"  event_type: {evt.event}")
                log(f"  inner_type: {evt.inner_type}")
                log(f"  block_type: {evt.block_type}")
                log(f"  delta_type: {evt.delta_type}")
                log(f"  text: {repr(evt.text[:300]) if evt.text else '(empty)'}")
                
                if isinstance(evt.data, dict):
                    log(f"  data_keys: {list(evt.data.keys())}")
                    if evt.event != "reason":
                        data_str = json.dumps(evt.data, ensure_ascii=False, indent=2)
                        if len(data_str) > 1000:
                            log(f"  data (truncated): {data_str[:1000]}...")
                        else:
                            log(f"  data: {data_str}")
                    else:
                        inner_evt = evt.data.get("event", {})
                        if isinstance(inner_evt, dict):
                            log(f"  inner_event_keys: {list(inner_evt.keys())}")
                            log(f"  inner_event: {json.dumps(inner_evt, ensure_ascii=False)[:500]}")
                        for key in ['tool_use', 'tool_name', 'action', 'status', 'stage', 'step', 'type', 'message', 'tool_use_result']:
                            if key in evt.data:
                                val_str = json.dumps(evt.data[key], ensure_ascii=False) if isinstance(evt.data[key], (dict, list)) else repr(evt.data[key])
                                log(f"  data.{key}: {val_str[:500]}")
                log()
        except Exception as e:
            log(f"\nERROR: {type(e).__name__}: {e}")
        
        log(f"\nTotal events: {count}")
        
        # Write to file
        with open(OUTPUT_FILE, "w") as f:
            f.write("\n".join(lines))
        
        try:
            await client.delete_session(sid)
        except:
            pass

asyncio.run(main())
