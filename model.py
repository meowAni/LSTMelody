import torch
import torch.nn as nn
import torch.nn.functional as F

min_pitch = 21
max_pitch = 108
n_pitches = max_pitch - min_pitch + 1
sequence_length = 128
n_velocities = 128

class MusicGen(nn.Module):
  def __init__(self):
    super(MusicGen, self).__init__()
    self.lstm = nn.LSTM(input_size=4, hidden_size=128, batch_first=True)

    self.pitch_layer = nn.Linear(128, n_pitches)
    self.velocity_layer = nn.Linear(128, n_velocities)
    self.step_layer = nn.Linear(128, 1)
    self.duration_layer = nn.Linear(128, 1)

    self.relu = nn.ReLU()
  
  def forward(self, x, hidden=None):
    # x : (batch_size, sequence_length, 4 ) tensor
    # lstm_out : (batch_size, sequence_length, 128) tensor
    x, hidden = self.lstm(x, hidden)
    # Give last layer of lstm_out for the 3 fully connected layer
    # x : (batch_size, 128) tensor
    x_last = x[:, -1, :]
    
    pitch = self.pitch_layer(x_last)
    
    velocity = self.velocity_layer(x_last)

    step = self.step_layer(x_last)
    step = self.relu(step)

    duration = self.duration_layer(x_last)
    duration = self.relu(duration)
    
    # out : (batch_size , n_pitches + n_velocities + 1 + 1) tensor
    out = torch.cat([pitch, velocity, duration, step], dim=-1)
    return out, hidden