# Responsible for preprocessing all the files from the source and saving it in chunks in the destination path

import os
import pretty_midi
from tqdm.auto import tqdm
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import warnings
from math import sqrt, log10
import random

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

  tmp_dur_drums = 0.0
  tmp_dur_bass = 0.0
  tmp_dur_chords = 0.0
  tmp_dur_lead = 0.0

  tmp_dur_sq_drums = 0.0
  tmp_dur_sq_bass = 0.0
  tmp_dur_sq_chords = 0.0
  tmp_dur_sq_lead = 0.0

  tmp_step_drums = 0.0
  tmp_step_other = 0.0

  tmp_step_sq_drums = 0.0
  tmp_step_sq_other = 0.0
  
  tmp_notes_drums = 0
  tmp_notes_bass = 0
  tmp_notes_chords = 0
  tmp_notes_lead = 0
  rolls = []

  for path in tqdm(file_paths, desc="Processing files"):
    roll = create_roll(path)
    if roll is None or len(roll) == 0:
      continue
    rolls.append(roll)
    for ins_idx in range(4):
      ins_roll = roll[roll[:, 4] == ins_idx]
      if len(ins_roll) == 0:
        continue
      
      if (ins_idx == 0):
        tmp_dur_drums += ins_roll[:, 2].sum().item()
        tmp_dur_sq_drums += (ins_roll[:, 2] ** 2).sum().item()
        tmp_step_drums += ins_roll[:, 3].sum().item()
        tmp_step_sq_drums += (ins_roll[:, 3] ** 2).sum().item()
        tmp_notes_drums += len(ins_roll)
      elif (ins_idx == 1):
        tmp_dur_bass += ins_roll[:, 2].sum().item()
        tmp_dur_sq_bass += (ins_roll[:, 2] ** 2).sum().item()
        tmp_step_other += ins_roll[:, 3].sum().item()
        tmp_step_sq_other += (ins_roll[:, 3] ** 2).sum().item()
        tmp_notes_bass += len(ins_roll)
      elif (ins_idx == 2):
        tmp_dur_chords += ins_roll[:, 2].sum().item()
        tmp_dur_sq_chords += (ins_roll[:, 2] ** 2).sum().item()
        tmp_step_other += ins_roll[:, 3].sum().item()
        tmp_step_sq_other += (ins_roll[:, 3] ** 2).sum().item()
        tmp_notes_chords += len(ins_roll)
      elif (ins_idx == 3):
        tmp_dur_lead += ins_roll[:, 2].sum().item()
        tmp_dur_sq_lead += (ins_roll[:, 2] ** 2).sum().item()
        tmp_step_other += ins_roll[:, 3].sum().item()
        tmp_step_sq_other += (ins_roll[:, 3] ** 2).sum().item()
        tmp_notes_lead += len(ins_roll)
  
  return rolls, tmp_dur_drums, tmp_dur_bass, tmp_dur_chords, tmp_dur_lead, tmp_dur_sq_drums, tmp_dur_sq_bass, tmp_dur_sq_chords, tmp_dur_sq_lead, tmp_step_drums, tmp_step_other, tmp_step_sq_drums, tmp_step_sq_other, tmp_notes_drums, tmp_notes_bass, tmp_notes_chords, tmp_notes_lead

def pad_num(n, width):
  return str(n).zfill(width)

pad_size = 1 # initial value

# takes a list of preprocessed data and saves it
def save_chunk(rolls, idx, parent, prefix):
  padded_num = pad_num(idx, int(pad_size))
  torch.save(rolls, os.path.join(parent, f"{prefix}_{padded_num}.pth"))

def dataproc(prefix):
  root = os.path.dirname(os.path.abspath(__file__))
  dest_path = os.path.join(root, Config.get("preprocessing")['processed_data_path'], prefix)
  src_path = os.path.join(root, Config.get('preprocessing')['raw_data_path'])
  file_paths = list(Path(src_path).rglob("*.mid"))
  n_files = Config.get("preprocessing")['n_files'] or len(file_paths)
  file_paths = random.sample(file_paths, n_files)
  shard_size = Config.get("preprocessing")['shard_size']

  print(f"Processing {min(n_files, len(file_paths))} files from {src_path}")
  
  # Normalizing coefficients

  global_dur_drums = 0.0
  global_dur_bass = 0.0
  global_dur_chords = 0.0
  global_dur_lead = 0.0
  
  global_dur_sq_drums = 0.0
  global_dur_sq_bass = 0.0
  global_dur_sq_chords = 0.0
  global_dur_sq_lead = 0.0
  
  global_step_drums = 0.0
  global_step_other = 0.0

  global_step_sq_drums = 0.0
  global_step_sq_other = 0.0

  global_notes_drums = 0
  global_notes_bass = 0
  global_notes_chords = 0
  global_notes_lead = 0

  chunks_processed = 0
  all_rel_idxs = []
  total_songs_processed = 0  # track total songs processed up to each chunk
  
  n_shards = round(n_files / shard_size)
  global pad_size
  pad_size = log10(n_shards) + 1

  for shard_idx in tqdm(range(0, n_files, shard_size), desc="Processing chunks"):
    shard_paths = file_paths[shard_idx:shard_idx + shard_size]
    if not shard_paths:
      continue
    
    result = process_files(shard_paths)
    if not result:
      continue
    rolls, \
    tmp_dur_drums, tmp_dur_bass, tmp_dur_chords, tmp_dur_lead, \
    tmp_dur_sq_drums, tmp_dur_sq_bass, tmp_dur_sq_chords, tmp_dur_sq_lead, \
    tmp_step_drums, tmp_step_other, \
    tmp_step_sq_drums, tmp_step_sq_other, \
    tmp_notes_drums, tmp_notes_bass, tmp_notes_chords, tmp_notes_lead = result
    
    save_chunk(rolls, chunks_processed, dest_path, prefix)
    
    # update total songs processed
    total_songs_processed += len(rolls)
    all_rel_idxs.append(total_songs_processed - 1)  # latest song index in this chunk
    
    global_dur_drums += tmp_dur_drums
    global_dur_bass += tmp_dur_bass
    global_dur_chords += tmp_dur_chords
    global_dur_lead += tmp_dur_lead
    global_dur_sq_drums += tmp_dur_sq_drums
    global_dur_sq_bass += tmp_dur_sq_bass
    global_dur_sq_chords += tmp_dur_sq_chords
    global_dur_sq_lead += tmp_dur_sq_lead
    global_step_drums += tmp_step_drums
    global_step_other += tmp_step_other
    global_step_sq_drums += tmp_step_sq_drums
    global_step_sq_other += tmp_step_sq_other
    global_notes_drums += tmp_notes_drums
    global_notes_bass += tmp_notes_bass
    global_notes_chords += tmp_notes_chords
    global_notes_lead += tmp_notes_lead
    chunks_processed += 1

  global_notes = global_notes_drums + global_notes_bass + global_notes_chords + global_notes_lead

  # Doing max with 0 inside sqrt to handle cases we get negative numbers due to slight rounding errors

  mean_dur_drums = global_dur_drums / global_notes_drums if global_notes_drums > 0 else 0.0
  mean_dur_bass = global_dur_bass / global_notes_bass if global_notes_bass > 0 else 0.0
  mean_dur_chords = global_dur_chords / global_notes_chords if global_notes_chords > 0 else 0.0
  mean_dur_lead = global_dur_lead / global_notes_lead if global_notes_lead > 0 else 0.0
  std_dur_drums = sqrt(max((global_dur_sq_drums / global_notes_drums) - (mean_dur_drums ** 2), 0.001)) if global_notes_drums > 0 else 0.001
  std_dur_bass = sqrt(max((global_dur_sq_bass / global_notes_bass) - (mean_dur_bass ** 2), 0.001)) if global_notes_bass > 0 else 0.001
  std_dur_chords = sqrt(max((global_dur_sq_chords / global_notes_chords) - (mean_dur_chords ** 2), 0.001)) if global_notes_chords > 0 else 0.001
  std_dur_lead = sqrt(max((global_dur_sq_lead / global_notes_lead) - (mean_dur_lead ** 2), 0.001)) if global_notes_lead > 0 else 0.001
  mean_step_drums = global_step_drums / global_notes_drums if global_notes_drums > 0 else 0.0
  mean_step_other = global_step_other / (global_notes_bass + global_notes_chords + global_notes_lead) if (global_notes_bass + global_notes_chords + global_notes_lead) > 0 else 0.0
  std_step_drums = sqrt(max((global_step_sq_drums / global_notes_drums) - (mean_step_drums ** 2), 0.001)) if global_notes_drums > 0 else 0.001
  std_step_other = sqrt(max((global_step_sq_other / (global_notes_bass + global_notes_chords + global_notes_lead)) - (mean_step_other ** 2), 0.001)) if (global_notes_bass + global_notes_chords + global_notes_lead) > 0 else 0.001

  torch.save({
    'rel_idxs': all_rel_idxs,

    'mean_duration_drums': mean_dur_drums,
    'mean_duration_bass': mean_dur_bass,
    'mean_duration_chords': mean_dur_chords,
    'mean_duration_lead': mean_dur_lead,

    'std_duration_drums': std_dur_drums,
    'std_duration_bass': std_dur_bass,
    'std_duration_chords': std_dur_chords,
    'std_duration_lead': std_dur_lead,

    'mean_step_drums': mean_step_drums,
    'mean_step_other': mean_step_other,

    'std_step_drums': std_step_drums,
    'std_step_other': std_step_other
  }, os.path.join(dest_path, f"{prefix}_meta.pth"))
  
  print(f"Processed {global_notes} notes across {chunks_processed} chunks")
  print(f"Data saved to {dest_path}")

if __name__ == "__main__":
  prefix = input("Enter folder name to save in: ")
  root = os.path.dirname(os.path.abspath(__file__))
  dest_path = os.path.join(root, Config.get("preprocessing")['processed_data_path'])
  os.makedirs(os.path.join(dest_path, prefix), exist_ok=True)
  dataproc(prefix)