import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import pretty_midi
import sys

min_drum_pitch = 35
max_drum_pitch = 81
n_drum_pitches = max_drum_pitch - min_drum_pitch + 1

min_pitch = 0
max_pitch = 127
n_pitches = max_pitch - min_pitch + 1

sequence_length = 128
n_velocities = 128
n_instruments = 4

def generate(model, seed_sequence, steps=512, device='cpu'):
  model.eval()
  # seed_sequence: (1, 128, 5)
  seed_sequence = seed_sequence.to(device)
  # generated_sequence: (steps, 5)
  generated_sequence = []
  hidden = None

  with torch.no_grad():
    for _ in tqdm(range(steps)):
      out, hidden = model(seed_sequence, hidden)
      instrument = torch.argmax(out['instrument'], dim=-1).item()
      drum_pitch = torch.multinomial(F.softmax(out['drum_pitch'], dim=-1), 1).item() + min_drum_pitch
      regular_pitch = torch.multinomial(F.softmax(out['regular_pitch'], dim=-1), 1).item()
      velocity = torch.multinomial(F.softmax(out['velocity'], dim=-1), 1).item()
      step = out['step'].item()
      duration = out['duration'].item()
      generated_note = torch.tensor([0, velocity, duration, step, instrument], device=device)
      if (instrument == 0):
        generated_note[0] = drum_pitch
      else:
        generated_note[0] = regular_pitch
      generated_sequence.append(generated_note)
      
      # newnote: (1, 1, 5) float32
      seed_sequence = torch.cat([seed_sequence[:, 1:, :], generated_note.unsqueeze(0).unsqueeze(0)], dim=1)

    generated_sequence = torch.stack(generated_sequence, dim=0)
    return generated_sequence
  
def tensor_to_midi(seqs):
  CATEGORY_PROGRAMS = {
    0: (0, True),   # Drums (is_drum=True)
    1: (32, False), # Bass (Acoustic Bass, program 32)
    2: (0, False),  # Chords (Acoustic Grand Piano, program 0)
    3: (56, False)  # Lead (Trumpet, program 56)
  }

  midi = pretty_midi.PrettyMIDI()
  instruments = {
    i: pretty_midi.Instrument(program=CATEGORY_PROGRAMS[i][0], is_drum=CATEGORY_PROGRAMS[i][1])
    for i in range(4)  # 0: Drums, 1: Bass, 2: Chords, 3: Lead
  }
  
  current_time = 0.0
  for i in range(seqs.shape[0]):
    pitch = int(seqs[i, 0].item())
    velocity = int(seqs[i, 1].item())
    duration = float(seqs[i, 2].item())
    step = float(seqs[i, 3].item())
    instrument = int(seqs[i, 4].item())
    current_time += step
    end_time = current_time + duration
    midi_note = pretty_midi.Note(
      velocity=int(velocity),
      pitch=int(pitch),
      start=current_time,
      end=end_time
    )
    
    instruments[instrument].notes.append(midi_note)

  for instr in instruments.values():
    midi.instruments.append(instr)

  return midi