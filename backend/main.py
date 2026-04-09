import asyncio
from pathlib import Path
from typing import Any, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.eeg_pipeline import CHUNK_SAMPLES, CLASSES, SAMPLING_RATE, CHANNELS, StreamRuntime


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
CHECKPOINT_PATH = BASE_DIR / "backend" / "model_checkpoint.json"


class StartRequest(BaseModel):
    seed: int | None = None


app = FastAPI(title="BCI Motor Imagery MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime = StreamRuntime(CHECKPOINT_PATH)
connections: Set[WebSocket] = set()


def class_payload() -> Dict[str, Any]:
    return {
        "classes": runtime.classifier.classes,
        "trainedClasses": runtime.classifier.trained_classes,
    }


async def broadcast(payload: Dict[str, Any]) -> None:
    stale: list[WebSocket] = []
    for socket in connections:
        try:
            await socket.send_json(payload)
        except Exception:
            stale.append(socket)

    for socket in stale:
        try:
            connections.remove(socket)
        except KeyError:
            pass


async def stream_loop() -> None:
    while True:
        if runtime.running:
            chunk, quality = runtime.next_frame()

            frame_event = {
                "type": "eeg_frame",
                "targetClass": runtime.target_class,
                "sampleRate": SAMPLING_RATE,
                "chunkSamples": CHUNK_SAMPLES,
                "quality": quality,
                "channels": CHANNELS,
                "samples": chunk.tolist(),
            }
            await broadcast(frame_event)

            if runtime.has_full_window():
                result = runtime.predict_latest()
                prediction_event = {
                    "type": "prediction",
                    "predictedClass": result.predicted_class,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "latencyMs": result.latency_ms,
                }
                await broadcast(prediction_event)

        await asyncio.sleep(CHUNK_SAMPLES / SAMPLING_RATE)


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(stream_loop())


@app.get("/health")
async def health() -> Dict[str, Any]:
    classes = class_payload()
    return {
        "status": "ok",
        "inferenceMode": runtime.inference_mode,
        "running": runtime.running,
        "targetClass": runtime.target_class,
        "classes": classes["classes"],
        "trainedClasses": classes["trainedClasses"],
    }


@app.get("/api/config")
async def config() -> Dict[str, Any]:
    classes = class_payload()
    return {
        "sampleRate": SAMPLING_RATE,
        "chunkSamples": CHUNK_SAMPLES,
        "channels": CHANNELS,
        "classes": classes["classes"],
        "trainedClasses": classes["trainedClasses"],
    }


@app.post("/api/session/start")
async def start_session(payload: StartRequest) -> Dict[str, Any]:
    runtime.start(payload.seed)
    await broadcast({"type": "status", "running": True})
    return {"ok": True, "running": runtime.running}


@app.post("/api/session/stop")
async def stop_session() -> Dict[str, Any]:
    runtime.stop()
    await broadcast({"type": "status", "running": False})
    return {"ok": True, "running": runtime.running}


@app.post("/api/session/class/{target_class}")
async def set_class(target_class: str) -> Dict[str, Any]:
    classes = class_payload()
    if target_class not in classes["classes"]:
        return {"ok": False, "message": "Invalid class", "classes": classes["classes"]}

    runtime.set_target_class(target_class)
    await broadcast({"type": "target_class", "targetClass": runtime.target_class})
    return {"ok": True, "targetClass": runtime.target_class}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    connections.add(websocket)
    await websocket.send_json(
        {
            "type": "hello",
            "running": runtime.running,
            "targetClass": runtime.target_class,
            "classes": runtime.classifier.classes,
            "trainedClasses": runtime.classifier.trained_classes,
            "channels": CHANNELS,
            "sampleRate": SAMPLING_RATE,
        }
    )

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
