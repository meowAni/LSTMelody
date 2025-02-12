# Responsible for preprocessing all the files from the source and saving it in chunks in the destination path

import os
import pretty_midi
from tqdm.auto import tqdm
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import warnings
from math import ceil, sqrt

from parse_config import Config

# return what an instrument be simplified to
def categorize_instrument(instrument):
  if instrument.is_drum:
    return "Drums"
  prog = instrument.program
  if 32 <= prog < 40:
    return "Bass"
  if 80 <= prog < 88:
    return "Lead"
  if 40 <= prog < 48:
    return "Lead" if "violin" in instrument.name.lower() else "Chords"
  if (0 <= prog < 8) or (16 <= prog < 24) or (24 <= prog < 32) or (88 <= prog < 96):
    return "Chords"
  return "Chords"

CATEGORY_PROGRAMS = {
  "Drums": (0, True), # No Program
  "Bass": (33, False), # Acoustic Bass
  "Chords": (0, False), # Acoustic Grand Piano
  "Lead": (56, False) # Trumpet
}

# return a new midi object after merging similar instruments
def merge_instruments(midi_obj):
  merged_tracks = {cat: None for cat in CATEGORY_PROGRAMS}
  CATEGORY_ORDER = ["Drums", "Bass", "Chords", "Lead"]

  for instrument in midi_obj.instruments:
    category = categorize_instrument(instrument)
    if category not in CATEGORY_ORDER:
      continue
    if merged_tracks[category] is None:
      program, is_drum = CATEGORY_PROGRAMS[category]
      merged_tracks[category] = pretty_midi.Instrument(
        program=program, is_drum=is_drum, name=category)
    merged_tracks[category].notes.extend(instrument.notes)

  new_midi = pretty_midi.PrettyMIDI()
  for cat in CATEGORY_ORDER:
    if merged_tracks[cat] is not None:
      new_midi.instruments.append(merged_tracks[cat])
    else:
      program, is_drum = CATEGORY_PROGRAMS[cat]
      new_midi.instruments.append(
        pretty_midi.Instrument(program=program, is_drum=is_drum, name=cat))
  return new_midi

# converts a midi file into a tensor
# each note consists of pitch, velocity, duration, step, instrument(0:drum, 1:bass, 2:chords, 3:lead)
def create_roll(midi_file):
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)
      midi = pretty_midi.PrettyMIDI(str(midi_file))
  except:
    return None

  midi = merge_instruments(midi)
  note_seq_list = []

  for ins_idx, instrument in enumerate(midi.instruments[:4]):
    for note in instrument.notes:
      if ins_idx == 0:  # Drums
        pitch = max(35, min(note.pitch, 81))
      else:
        pitch = note.pitch
      note_seq_list.append((
        pitch,
        note.velocity,
        note.end - note.start,
        note.start,
        ins_idx  # Correct index based on CATEGORY_ORDER
      ))

  if not note_seq_list:
    return None

  note_seq_arr = np.array(note_seq_list, dtype=np.float32)
  note_seq_arr = note_seq_arr[np.argsort(note_seq_arr[:, 3])]
  note_seq_arr[1:, 3] -= note_seq_arr[:-1, 3].copy()
  note_seq_arr[0, 3] = 0.0
  return torch.from_numpy(note_seq_arr)

# takes a list of files and preprocesses it
def process_files(file_paths):
  total_dur, total_step = 0.0, 0.0
  total_dur_sq, total_step_sq = 0.0, 0.0
  total_notes = 0
  rolls = []

  for path in tqdm(file_paths, desc="Processing files"):
    roll = create_roll(path)
    if roll is None or len(roll) == 0:
      continue
    
    durations = roll[:, 2]
    steps = roll[:, 3]
    n_notes = len(durations)
    
    total_dur += durations.sum().item()
    total_step += steps.sum().item()
    total_dur_sq += (durations ** 2).sum().item()
    total_step_sq += (steps ** 2).sum().item()
    total_notes += n_notes
    rolls.append(roll)
  
  return rolls, total_dur, total_step, total_dur_sq, total_step_sq, total_notes

# takes a list of preprocessed files and saves it
def save_chunk(rolls, idx, parent, prefix):
  torch.save(rolls, os.path.join(parent, f"{prefix}_{idx}.pth"))

def dataproc(prefix):
  root = os.path.dirname(os.path.abspath(__file__))
  dest_path = os.path.join(root, Config.get("preprocessing")['processed_data_path'], prefix)
  src_path = os.path.join(root, Config.get('preprocessing')['raw_data_path'])
  n_files = Config.get("preprocessing")['n_files'] or len(file_paths)
  file_paths = list(Path(src_path).rglob("*.mid"))
  file_paths = file_paths[:n_files]
  chunk_size = Config.get("preprocessing")['chunk_size']

  print(f"Processing {min(n_files, len(file_paths))} files from {src_path}")

  global_dur = global_step = 0.0
  global_dur_sq = global_step_sq = 0.0
  global_notes = 0
  chunks_processed = 0
  all_rel_idxs = []
  total_songs_processed = 0  # track total songs processed up to each chunk
  
  for chunk_idx in tqdm(range(0, n_files, chunk_size), desc="Processing chunks"):
    chunk_paths = file_paths[chunk_idx:chunk_idx + chunk_size]
    if not chunk_paths:
      continue
    
    result = process_files(chunk_paths)
    if not result:
      continue
    rolls, td, ts, tds, tss, tn = result
    if tn == 0:
      continue
    
    save_chunk(rolls, chunks_processed, dest_path, prefix)
    
    # update total songs processed
    total_songs_processed += len(rolls)
    all_rel_idxs.append(total_songs_processed - 1)  # latest song index in this chunk
    
    # accumulate global stats
    global_dur += td
    global_step += ts
    global_dur_sq += tds
    global_step_sq += tss
    global_notes += tn
    chunks_processed += 1

  if global_notes > 0:
    mean_duration = global_dur / global_notes
    mean_step = global_step / global_notes
    var_duration = (global_dur_sq / global_notes) - (mean_duration ** 2)
    var_step = (global_step_sq / global_notes) - (mean_step ** 2)
    std_duration = sqrt(var_duration)
    std_step = sqrt(var_step)
  else:
    mean_duration = mean_step = std_duration = std_step = 0.0

  torch.save({
    'rel_idxs': all_rel_idxs,
    'mean_duration': mean_duration,
    'mean_step': mean_step,
    'std_duration': std_duration,
    'std_step': std_step
  }, os.path.join(dest_path, f"{prefix}_meta.pth"))
  
  print(f"Processed {global_notes} notes across {chunks_processed} chunks")
  print(f"Data saved to {dest_path}")

if __name__ == "__main__":
  prefix = input("Enter folder name to save in: ")
  root = os.path.dirname(os.path.abspath(__file__))
  dest_path = os.path.join(root, Config.get("preprocessing")['processed_data_path'])
  os.makedirs(os.path.join(dest_path, prefix), exist_ok=True)
  dataproc(prefix)