from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import asyncio
import redis.asyncio as aioredis
from typing import Dict
from .models import JobPayload
from .worker import process_batch_job

app = FastAPI(title="MZIX Server API")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_telemetry(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)

manager = ConnectionManager()

@app.post("/api/v1/jobs")
async def submit_job(payload: JobPayload):
    # Dispatch to Celery queue, mapping FastAPI priority to Celery priority
    # Note: payload.priority is used to order the queue.
    task = process_batch_job.apply_async(args=[payload.model_dump()], priority=payload.priority)
    return {"status": "queued", "task_id": task.id, "job_id": payload.job_id}

@app.websocket("/ws/telemetry/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    
    # Connect to Redis PubSub to listen for telemetry from the Celery worker
    redis = await aioredis.from_url("redis://localhost:6379/0")
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"telemetry:{user_id}")
    
    try:
        # A background task that reads from Redis and sends via WebSocket
        async def redis_reader():
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    data = json.loads(message["data"].decode("utf-8"))
                    await manager.send_telemetry(user_id, data)
                await asyncio.sleep(0.01)

        reader_task = asyncio.create_task(redis_reader())
        
        while True:
            # Keep connection alive, wait for client pings
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user_id)
        reader_task.cancel()
        await pubsub.unsubscribe(f"telemetry:{user_id}")
        await redis.close()
