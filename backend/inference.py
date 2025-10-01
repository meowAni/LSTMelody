import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import pretty_midi

min_drums_pitch = 35
max_drums_pitch = 81
n_drums_pitches = max_drums_pitch - min_drums_pitch + 1

min_pitch = 0
max_pitch = 127
n_pitches = max_pitch - min_pitch + 1

sequence_length = 128
n_velocities = 128
n_instruments = 4

def generate(model, seed_sequence, prob_instrument_drums, prob_instrument_bass, prob_instrument_chords, prob_instrument_lead, steps=512, device='cpu'):
  model.eval()
  # seed_sequence: (1, 128, 5)
  seed_sequence = seed_sequence.to(device)
  # generated_sequence: (steps, 5)
  generated_sequence = []
  hidden = None

  instrument_probs = torch.tensor([[prob_instrument_drums, prob_instrument_bass, prob_instrument_chords, prob_instrument_lead]]).to(device)
  with torch.no_grad():
    for _ in tqdm(range(steps)):
      out, hidden = model(seed_sequence, hidden)

      for k, v in out.items():
          if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"Invalid output detected in {k}: {v}")
      
      instrument = F.softmax(out["instrument"], dim=-1)
      instrument = instrument * instrument_probs
      instrument = torch.multinomial(instrument, 1).item()

      pitch_drums = torch.multinomial(F.softmax(out["pitch_drums"], dim=-1), 1).item() + min_drums_pitch
      pitch_bass = torch.multinomial(F.softmax(out["pitch_bass"], dim=-1), 1).item()
      pitch_chords = torch.multinomial(F.softmax(out["pitch_chords"], dim=-1), 1).item()
      pitch_lead = torch.multinomial(F.softmax(out["pitch_lead"], dim=-1), 1).item()
      
      velocity_drums = torch.multinomial(F.softmax(out["velocity_drums"], dim=-1), 1).item()
      velocity_other = torch.multinomial(F.softmax(out["velocity_other"], dim=-1), 1).item()

      duration_drums = out["duration_drums"].item()
      duration_bass = out["duration_bass"].item()
      duration_chords = out["duration_chords"].item()
      duration_lead = out["duration_lead"].item()

      step_drums = out["step_drums"].item()
      step_other = out["step_other"].item()

      if (step_drums < 0.007):
        step_drums = 0.007
      if (step_other < 0.007):
        step_other = 0.007

      generated_note = torch.tensor([0.0, 0.0, 0.0, 0.0, instrument], device=device)

      if (instrument == 0): # Drums
        generated_note[0] = pitch_drums
        generated_note[1] = velocity_drums
        generated_note[2] = duration_drums
        generated_note[3] = step_drums
      elif (instrument == 1): # Bass
        generated_note[0] = pitch_bass
        generated_note[1] = velocity_other
        generated_note[2] = duration_bass
        generated_note[3] = step_other
      elif (instrument == 2): # Chords
        generated_note[0] = pitch_chords
        generated_note[1] = velocity_other
        generated_note[2] = duration_chords
        generated_note[3] = step_other
      elif (instrument == 3): # Lead
        generated_note[0] = pitch_lead
        generated_note[1] = velocity_other
        generated_note[2] = duration_lead
        generated_note[3] = step_other
        
      generated_sequence.append(generated_note)
      
      # newnote: (1, 1, 5) float32
      seed_sequence = torch.cat([seed_sequence[:, 1:, :], generated_note.unsqueeze(0).unsqueeze(0)], dim=1)

    generated_sequence = torch.stack(generated_sequence, dim=0)
    return generated_sequence
  
def tensor_to_midi(seqs):
  CATEGORY_PROGRAMS = {
    0: (0, True),   # Drums (is_drums=True)
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
      velocity = int(velocity),
      pitch = int(pitch),
      start = current_time,
      end = end_time
    )
    
    instruments[instrument].notes.append(midi_note)

  for instr in instruments.values():
    midi.instruments.append(instr)

  return midi