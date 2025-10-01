from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel  
from uuid import uuid4
from os import path, makedirs
from createMusic import postprocess
from starlette.concurrency import run_in_threadpool
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
    instrumentDrums: int
    instrumentBass: int
    instrumentChords: int
    instrumentLead: int


@api.post("/create")
async def create_music(data: MusicRequest):
    seed_value = data.seed
    duration_value = data.duration
    instrument_drums = data.instrumentDrums
    instrument_bass = data.instrumentBass
    instrument_chords = data.instrumentChords
    instrument_lead = data.instrumentLead
    music_id = str(uuid4())
    print(f"[{music_id}] Request with seed = {seed_value}, duration = {duration_value}, drums = {instrument_drums}, bass = {instrument_bass}, chords = {instrument_chords}, lead = {instrument_lead}")
    music_path = path.join(base_dir, music_id)
    makedirs(music_path, exist_ok=True)
    await run_in_threadpool(
        postprocess, music_id, seed_value, duration_value, instrument_drums, instrument_bass, instrument_chords, instrument_lead, music_path
    )
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