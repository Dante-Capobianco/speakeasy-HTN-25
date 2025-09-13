from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class VideoPayload(BaseModel):
    videoUrl: str

@app.post("/")
async def root(payload: VideoPayload):
    print(payload.videoUrl)
    return payload.videoUrl