# 📄 LSTMelody

## 💻 Overview

**LSTMelody** is a deep learning project that generates multi-instrument MIDI songs, including drums, bass, chords, and lead instruments. It has a custom LSTM architecture trained on a subset of the Lakh MIDI dataset and an efficient and scalable data pipeline for structured input. It consists of a user-friendly responsive interface built with React, and a FastAPI backend to perform model inference.

## 📚 Tech Stack

- **Frontend:** React  
- **Backend:** FastAPI
- **Machine Learning:** PyTorch

## ♣️ Features

- **Music Generation:** Users can create harmonic, multi-instrument songs with control over song length, starting seed, and instrument probability.

- **Music Gallery:** Generated music can be played back using integrated controls, downloaded as MIDI or WAV, and deleted directly from the gallery.

## 📸 Screenshots

- **Desktop view**:
<img src="examples/screenshots/desktop.png" width="100%" />

- **Mobile view**:
<p float="left">
  <img src="examples/screenshots/mobile1.png" width="40%" style="margin-right:5%;" />
  <img src="examples/screenshots/mobile2.png" width="40%" />
</p>


## 🔊 Music
Few sample generated tracks

- **Music 1** (drum = 1, bass = 4, chords = 1, lead = 2)
<audio controls>
  <source src="https://raw.githubusercontent.com/meowAni/LSTMelody/main/examples/music/music1-drum1-bass4-chords1-lead2.mp3" type="audio/mp3">
  Your browser does not support the audio element.
</audio>

- **Music 2** (drum = 1, bass = 5, chords = 1, lead = 0)
<audio controls>
  <source src="https://raw.githubusercontent.com/meowAni/LSTMelody/main/examples/music/music2-drum1-bass5-chords1-lead0.mp3" type="audio/mp3">
  Your browser does not support the audio element.
</audio>

- **Music 3** (drum = 1, bass = 3, chords = 1, lead = 3)
<audio controls>
  <source src="https://raw.githubusercontent.com/meowAni/LSTMelody/main/examples/music/music3-drum1-bass3-chords1-lead3.mp3" type="audio/mp3">
  Your browser does not support the audio element.
</audio>

## 👥 Contributors

- [Anirudh Vignesh](https://github.com/crystallyen)
- [Divyesh Dileep](https://github.com/Divyesh48960)