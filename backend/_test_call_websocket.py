"""Scratch script: send two real audio turns to /ws/calls/{call_id} using the
SAME call_id, to verify Redis-backed state accumulates correctly across turns
(turn_count increments, sentiment_history grows, escalation risk reflects the
worsening second sentence). Delete after use."""

import asyncio
import json

import websockets

CALL_ID = "test-call-001"
WS_URL = f"ws://localhost:8000/ws/calls/{CALL_ID}"
AUDIO_FILES = ["_test_audio_1.wav", "_test_audio_2.wav"]


async def main() -> None:
    async with websockets.connect(WS_URL) as ws:
        for i, filename in enumerate(AUDIO_FILES, start=1):
            with open(filename, "rb") as f:
                audio_bytes = f.read()

            print(f"\n--- Sending turn {i}: {filename} ---")
            await ws.send(audio_bytes)

            response = await ws.recv()
            data = json.loads(response)
            print(json.dumps(data, indent=2))

            if data.get("type") == "error":
                print(f"WARNING: turn {i} returned an error, check server logs")


if __name__ == "__main__":
    asyncio.run(main())