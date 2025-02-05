import torch
import torch.nn as nn
import torch.nn.functional as F

min_drum_pitch = 35
max_drum_pitch = 81
n_drum_pitches = max_drum_pitch - min_drum_pitch + 1

min_pitch = 0
max_pitch = 127
n_pitches = max_pitch - min_pitch + 1

sequence_length = 128
n_velocities = 128
n_instruments = 4

class MusicGen(nn.Module):
  def __init__(self):
    super(MusicGen, self).__init__()
    self.lstm = nn.LSTM(input_size=5, hidden_size=512, batch_first=True)

    self.regular_pitch_head = nn.Linear(512, n_pitches)
    self.drum_pitch_head = nn.Linear(512, n_drum_pitches)
    self.velocity_head = nn.Linear(512, n_velocities)
    self.step_head = nn.Linear(512, 1)
    self.duration_head = nn.Linear(512, 1)
    self.instrument_head = nn.Linear(512, n_instruments)
  
  def forward(self, x, hidden=None):
    # x : (batch_size, sequence_length, 4 ) tensor
    # lstm_out : (batch_size, sequence_length, 128) tensor
    x, hidden = self.lstm(x, hidden)
    # Give last layer of lstm_out for the 3 fully connected layer
    # x : (batch_size, 128) tensor
    x_last = x[:, -1, :]
    
    # predict instrument
    drum_pitch_logits = self.drum_pitch_head(x_last)
    regular_pitch_logits = self.regular_pitch_head(x_last)
    velocity_logits = self.velocity_head(x_last)
    step = self.step_head(x_last)
    duration = self.duration_head(x_last)
    instrument_logits = self.instrument_head(x_last)
    
    # out : (batch_size , ...) tensor
    return {
      "instrument": instrument_logits,
      "drum_pitch": drum_pitch_logits,
      "regular_pitch": regular_pitch_logits,
      "velocity": velocity_logits,
      "step": step,
      "duration": duration
    }, hidden