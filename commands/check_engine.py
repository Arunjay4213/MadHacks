"""Check command/state behavior against a running engine with --reader commands.

Run: python -m commands.check_engine --url http://127.0.0.1:8000
Uses the engine's existing websocket check helpers and dependencies.
"""

import argparse
import asyncio
import time

from contracts.validate import errors
from engine.tools.e2e_check import Client, wait_for


async def check(url):
    client = Client(url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/"))
    tasks = [asyncio.create_task(client.run_frames()), asyncio.create_task(client.run_control())]
    try:
        assert await wait_for(lambda: client.control is not None and client.frames, 10), "no stream"

        async def send(message, labels, unmatched=False):
            message["t_client"] = int(time.time() * 1000)
            start = len(client.states)
            await client.send(message)

            def received():
                for _, state in client.states[start:]:
                    if (state.get("heard") == message.get("text")
                            and [c["label"] for c in state["changes"]] == labels
                            and state["unmatched"] == unmatched):
                        return state
                return None

            state = await wait_for(received, 8)
            assert state, f"state not received for {message}"
            assert not errors(4, state), errors(4, state)
            assert not client.errors, client.errors
            print("PASS", message.get("text", message.get("action")), labels)

        await send(dict(type="tap", action="reset"), [])
        await send(dict(type="partial", text="make it winter"), [])
        await send(dict(type="final", text="make it nice"), [], unmatched=True)
        for text, labels in [
            ("make it winter", ["winter"]),
            ("also add a river", ["winter", "river"]),
            ("slowly turn to night", ["winter", "river", "night"]),
            ("remove the river", ["winter", "night"]),
            ("go back", ["winter", "river", "night"]),
        ]:
            await send(dict(type="final", text=text), labels)
        await send(dict(type="tap", action="reset"), [])
        await send(dict(type="tap", action="undo"), ["winter", "river", "night"])
        await send(dict(type="final", text="don’t add a rainbow"), ["winter", "river", "night"], unmatched=True)
        assert all(not errors(4, state) for _, state in client.states)
        assert len(client.frames) > 1, "frames stopped"
        print(f"PASSED: {len(client.frames)} streamed frames; all {len(client.states)} state messages validate")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    asyncio.run(check(parser.parse_args().url))
