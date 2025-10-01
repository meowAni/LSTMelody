import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from midi2audio import FluidSynth
from os import path
import torch
from model import MusicGen
from inference import generate, tensor_to_midi
import io
import soundfile as sf
import sys, os
from contextlib import contextmanager

@contextmanager # To hide all the fluidsynth messages
def suppress_fluidsynth_output():
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    devnull = open(os.devnull, 'w')
    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout_fd = os.dup(stdout_fd)
    old_stderr_fd = os.dup(stderr_fd)
    os.dup2(devnull.fileno(), stdout_fd)
    os.dup2(devnull.fileno(), stderr_fd)

    try:
        yield
    finally:
        os.dup2(old_stdout_fd, stdout_fd)
        os.dup2(old_stderr_fd, stderr_fd)
        devnull.close()
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

seeds = torch.load('seeds.pth', weights_only=False)
meta = torch.load('meta.pth', weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MusicGen(meta)
model.load_state_dict(torch.load('weights.pth', weights_only=False)['model'])
model.to(device)

soundfont_path = "./FluidR3_GM.sf2"
midi_converter = FluidSynth(soundfont_path)

def postprocess(music_id, seed_inp, duration_inp, instrument_drums, instrument_bass, instrument_chords, instrument_lead, out_dir):
  # create midi
  print(f'[{music_id}] 1/3 Generating Music')
  seed = seeds[seed_inp]
  pm_object = tensor_to_midi(generate(model, seed.unsqueeze(0), instrument_drums, instrument_bass, instrument_chords, instrument_lead, duration_inp, device))
  midi_dir = path.join(out_dir, 'music.midi')
  pm_object.write(midi_dir)

  # create wav
  print(f'[{music_id}] 2/3 Creating Wav')
  wav_path = path.join(out_dir, 'music.wav')
  with suppress_fluidsynth_output():
    midi_converter.midi_to_audio(midi_dir, wav_path)
  
  # trimming wav to actual midi end time because it overestimates slightly
  actual_midi_end_time = pm_object.get_end_time()
  y, sr = sf.read(wav_path)
  samples_to_keep = int((actual_midi_end_time + 0.1) * sr) # 0.1s buffer just in case
  if samples_to_keep > len(y): # incase out of bounds
      samples_to_keep = len(y)
  trimmed_y = y[:samples_to_keep]
  sf.write(wav_path, trimmed_y, sr)

  # create image
  print(f'[{music_id}] 3/3 Creating Image')
  y, sr = librosa.load(wav_path)
  S = librosa.feature.melspectrogram(y=y, sr=sr)
  S_db = librosa.power_to_db(S, ref=np.max)

  # plot without axes or colorbar
  fig, ax = plt.subplots(figsize=(2, 2), dpi=64)  # small size for pixelation
  librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap='cool', ax=ax)
  ax.axis('off')
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  # save to buffer
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
  plt.close(fig)

  # pixelate image
  buf.seek(0)
  img = Image.open(buf)
  img = img.resize((1028, 512), resample=Image.NEAREST)
  img.save(path.join(out_dir, "preview.png"))

if __name__ == "__main__":
  print("no")