import os
import time
import json
import random

import aiohttp
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Request, Header, Body
from pydantic import BaseModel, Field

from client.balancer import LoadBalancer, RoundRobinLoadBalancer
from client.async_openai import AsyncOpenAI
from client.client_utils import predict_price, build_base_url

# TODO: replace this yaml config with a real config file
MODEL_CONFIG = {
    "CohereForAI/aya-expanse-8b": [
        {"host": "node14", "port": 48880, "api_key": "sk-hynix"},
        {"host": "node14", "port": 48881, "api_key": "sk-hynix"}
    ]
}

load_balancers = {}
for model_name, endpoints in MODEL_CONFIG.items():
    load_balancers[model_name] = RoundRobinLoadBalancer(endpoints)

app = FastAPI()

@app.get("/models")
async def list_models():
    return {"models": list(MODEL_CONFIG.keys())}

@app.get("/health")
async def global_health():
    return {"status": "ok"}

@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def completions(request: Request, req: dict = Body(...)):
    model_name = req["model"]
    if model_name not in load_balancers:
        raise HTTPException(status_code=404, detail="Model not found")

    lb = load_balancers[model_name]
    endpoint = lb.get_endpoint()
    host, port = endpoint["host"], endpoint["port"]
    base_url = build_base_url(host=host, port=port)
    
    if 'gemini' in model_name:
        pass
    else:
        base_url = f"{base_url.rstrip('/')}/v1"
    client = AsyncOpenAI(base_url=base_url, api_key=endpoint["api_key"])
    
    path = request.url.path

    async with aiohttp.ClientSession() as session:
        elapsed = time.time()
        if path.endswith("chat/completions"):
            response, request_id = await client.request(session, f"{base_url}/chat/completions", {
                "model": model_name,
                **req 
            })
        else:
            response, request_id = await client.request(session, f"{base_url}/completions", {
                "model": model_name,
                **req 
            })
        elapsed = time.time() - elapsed
        status = response.get('status', 500)

        lb.update_metrics(endpoint, elapsed, success=(200 <= status < 300))

        # 에러 처리 - 상위 레벨에서 진행
        if status == 401:
            raise HTTPException(status_code=401, detail="Unauthorized")
        elif status == 429:
            raise HTTPException(status_code=429, detail="Too Many Requests")
        elif status >= 400:
            raise HTTPException(status_code=status, detail=f"Upstream error: {response}")

        return response, request_id

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1557)