import torch
import torch.nn as nn
import torch.nn.functional as F

min_drums_pitch = 35
max_drums_pitch = 81
n_drums_pitches = max_drums_pitch - min_drums_pitch + 1

min_pitch = 0
max_pitch = 127
n_pitches = max_pitch - min_pitch + 1

sequence_length = 128
n_velocities = 128
n_instruments = 4

n_pitch_embed = 64
n_velocity_embed = 32
n_instrument_embed = 8

input_size = n_pitch_embed + n_velocity_embed + n_instrument_embed + 1 + 1  # pitch, velocity, instrument, duration, step

class Residual(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.fc1 = nn.Linear(dim, dim)
    self.act = nn.GELU()
    self.dropout = nn.Dropout(0.1)
    self.fc2 = nn.Linear(dim, dim)

  def forward(self, x):
    out = self.fc1(x)
    out = self.act(out)
    out = self.dropout(out)
    out = self.fc2(out)
    return x + out

class MusicGen(nn.Module):
  def __init__(self, meta, hidden_size=512):
    super(MusicGen, self).__init__()

    self.pitch_embed = nn.Embedding(n_pitches, n_pitch_embed)
    self.velocity_embed = nn.Embedding(n_velocities, n_velocity_embed)
    self.instrument_embed = nn.Embedding(n_instruments, n_instrument_embed)

    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.1, batch_first=True)

    self.fc1 = Residual(dim=hidden_size)
    self.fc2 = Residual(dim=hidden_size)

    def classifier(dim, output_dim):
      return nn.Sequential(
        nn.Linear(dim, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, output_dim)
      )

    self.pitch_drums = classifier(hidden_size, n_drums_pitches)
    self.pitch_bass = classifier(hidden_size, n_pitches)
    self.pitch_chords = classifier(hidden_size, n_pitches)
    self.pitch_lead = classifier(hidden_size, n_pitches)

    self.velocity_drums = classifier(hidden_size, n_velocities)
    self.velocity_other = classifier(hidden_size, n_velocities)

    self.instrument_head = classifier(hidden_size, n_instruments)

    def regressor(dim):
      return nn.Sequential(
        nn.Linear(dim, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1)
      )

    self.duration_drums = regressor(hidden_size)
    self.duration_bass = regressor(hidden_size)
    self.duration_chords = regressor(hidden_size)
    self.duration_lead = regressor(hidden_size)

    self.step_drums = regressor(hidden_size)
    self.step_other = regressor(hidden_size)

    self.meta = meta
  
  def forward(self, x, hidden=None):
    # x : (batch_size, sequence_length, 5) tensor
    # Normalize duration and step using per-instrument stats
    # Instrument index is in x[:, :, 4]
    duration_normalized = torch.zeros_like(x[:, :, 2])
    step_normalized = torch.zeros_like(x[:, :, 3])

    for i in range(4):
      mask = (x[:, :, 4] == i)
      # Duration normalization
      if i == 0:
        duration_normalized[mask] = (x[:, :, 2][mask] - self.meta['mean_duration_drums']) / (self.meta['std_duration_drums'] + 1e-8)
        step_normalized[mask] = (x[:, :, 3][mask] - self.meta['mean_step_drums']) / (self.meta['std_step_drums'] + 1e-8)
      elif i == 1:
        duration_normalized[mask] = (x[:, :, 2][mask] - self.meta['mean_duration_bass']) / (self.meta['std_duration_bass'] + 1e-8)
        step_normalized[mask] = (x[:, :, 3][mask] - self.meta['mean_step_other']) / (self.meta['std_step_other'] + 1e-8)
      elif i == 2:
        duration_normalized[mask] = (x[:, :, 2][mask] - self.meta['mean_duration_chords']) / (self.meta['std_duration_chords'] + 1e-8)
        step_normalized[mask] = (x[:, :, 3][mask] - self.meta['mean_step_other']) / (self.meta['std_step_other'] + 1e-8)
      elif i == 3:
        duration_normalized[mask] = (x[:, :, 2][mask] - self.meta['mean_duration_lead']) / (self.meta['std_duration_lead'] + 1e-8)
        step_normalized[mask] = (x[:, :, 3][mask] - self.meta['mean_step_other']) / (self.meta['std_step_other'] + 1e-8)

    pitch_embed = self.pitch_embed(x[:, :, 0].long().clamp(0, n_pitches - 1))         # (B, T, n_pitch_embed)
    velocity_embed = self.velocity_embed(x[:, :, 1].long().clamp(0, n_velocities - 1)) # (B, T, n_velocity_embed)
    instrument_embed = self.instrument_embed(x[:, :, 4].long().clamp(0, n_instruments - 1)) # (B, T, n_instrument_embed)

    # Concatenate all the features
    x_input = torch.cat([
        pitch_embed,
        velocity_embed,
        duration_normalized.unsqueeze(-1),
        step_normalized.unsqueeze(-1),
        instrument_embed
    ], dim=-1)

    x_last, hidden = self.lstm(x_input, hidden)
    # Give last layer of lstm_out for the 3 fully connected layer
    # x : (batch_size, 256) tensor
    x_last = x_last[:, -1, :]
    
    x_last = self.fc1(x_last)
    x_last = self.fc2(x_last)

    # predict instrument
    pitch_drums_logits = self.pitch_drums(x_last)
    pitch_bass_logits = self.pitch_bass(x_last)
    pitch_chords_logits = self.pitch_chords(x_last)
    pitch_lead_logits = self.pitch_lead(x_last)

    velocity_drums_logits = self.velocity_drums(x_last)
    velocity_other_logits = self.velocity_other(x_last)

    duration_drums = self.duration_drums(x_last) * self.meta['std_duration_drums'] + self.meta['mean_duration_drums']
    duration_bass = self.duration_bass(x_last) * self.meta['std_duration_bass'] + self.meta['mean_duration_bass']
    duration_chords = self.duration_chords(x_last) * self.meta['std_duration_chords'] + self.meta['mean_duration_chords']
    duration_lead = self.duration_lead(x_last) * self.meta['std_duration_lead'] + self.meta['mean_duration_lead']

    step_drums = self.step_drums(x_last) * self.meta['std_step_drums'] + self.meta['mean_step_drums']
    step_other = self.step_other(x_last) * self.meta['std_step_other'] + self.meta['mean_step_other']

    instrument_logits = self.instrument_head(x_last)

    # out : (batch_size , ...) tensor
    return {
      "instrument": instrument_logits,

      "pitch_drums": pitch_drums_logits,
      "pitch_bass": pitch_bass_logits,
      "pitch_chords": pitch_chords_logits,
      "pitch_lead": pitch_lead_logits,

      "velocity_drums": velocity_drums_logits,
      "velocity_other": velocity_other_logits,

      "duration_drums": duration_drums,
      "duration_bass": duration_bass,
      "duration_chords": duration_chords,
      "duration_lead": duration_lead,

      "step_drums": step_drums,
      "step_other": step_other
    }, hidden