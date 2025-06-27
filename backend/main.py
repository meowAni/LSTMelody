from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel  
from uuid import uuid4
from os import path, makedirs
from createMusic import postprocess
import shutil

base_dir = path.abspath("musics")
src_path = path.join(base_dir, 'source.mid')

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicRequest(BaseModel):
    seed: int
    duration: int

@api.post("/create")
def create_music(data: MusicRequest):
    seed_value = data.seed
    duration_value = data.duration
    print(f"Recieved create request with seed {seed_value} and duration {duration_value}")
    music_id = str(uuid4())
    music_path = path.join(base_dir, music_id)
    makedirs(music_path, exist_ok=True)
    postprocess(seed_value, duration_value, music_path)
    return {
        "id": music_id,
        "img": f"http://localhost:8000/musics/{music_id}/preview.png",
        "mid": "http://localhost:8000/musics/source.mid",
        "wav": f"http://localhost:8000/musics/{music_id}/music.wav"
    }

@api.delete("/delete/{uuid}")
def delete_music(uuid : str):
    folder = path.join(base_dir, uuid)
    if path.exists(folder):
      shutil.rmtree(folder)

api.mount("/musics", StaticFiles(directory="musics"), name="musics")