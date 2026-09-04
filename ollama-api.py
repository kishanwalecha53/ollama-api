import asyncio
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/infer")
async def infer(request: Request):
    payload = await request.json()

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()

        async def ollama_reader():
            try:
                async with httpx.AsyncClient(timeout=1200) as client:
                    async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line:
                                await queue.put(line)
            except Exception as e:
                await queue.put(json.dumps({"Exception caught while executing the task on ec2": str(e)}))
            finally:
                await queue.put(None)  

        asyncio.create_task(ollama_reader())

        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=10.0)
                if line is None:
                    break
                yield line + "\n"
            except asyncio.TimeoutError:
                yield '{"keepalive":true}\n'

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    return {"status": "ok"}
