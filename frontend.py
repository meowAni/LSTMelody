import torch
import streamlit as st
import tempfile
import os
from model import MusicGen
from inference import generate, tensor_to_midi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seeds = torch.load('seeds.pth', weights_only=False)
meta = torch.load('meta.pth', weights_only=False)
model = MusicGen(meta)
model.load_state_dict(torch.load('trained_weights/model12feb-model', weights_only=True))
model.to(device)

def get_midi_bytes(pm_object):
    # write the PrettyMIDI object to a temporary file and return its bytes for download.
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as midi_file:
        pm_object.write(midi_file.name)
        midi_path = midi_file.name
    with open(midi_path, 'rb') as f:
        midi_bytes = f.read()
    os.remove(midi_path)
    return midi_bytes

if 'songs' not in st.session_state:
    st.session_state['songs'] = []  # list of songs where each song is a dict and has id, audio bytes, and midi bytes.
if 'song_counter' not in st.session_state:
    st.session_state['song_counter'] = 0

# App layout
st.title("AI Music Generator")

# Input
col1, col2 = st.columns(2)
with col1:
    num_notes = st.number_input("Number of Notes", min_value=100, max_value=10000, value=100, step=1)
with col2:
    seed_choice = st.selectbox("Select Seed", options=[f"Seed {i}" for i in range(1, 11)])

if st.button("Generate Music"):
    seed_index = int(seed_choice.split(" ")[1]) - 1
    seed = seeds[seed_index]
    pm_object = tensor_to_midi(generate(model, seed.unsqueeze(0), num_notes, device))
    midi_bytes = get_midi_bytes(pm_object)
    
    # store the song with a unique id
    song_id = st.session_state['song_counter']
    st.session_state['songs'].append({
        'id': song_id,
        'pm_object': pm_object,
        'midi_bytes': midi_bytes
    })
    st.session_state['song_counter'] += 1
    st.success("Music generated!")

st.markdown("---")
st.header("Gallery")

cols = st.columns(5)
for idx, song in enumerate(st.session_state['songs']):
    with cols[idx % 5]:
        st.subheader(f"Song {song['id']}")
        
        st.download_button(
            label="🎵 Download",
            data=song['midi_bytes'],
            file_name=f"song_{song['id']}.mid",
            mime="audio/midi"
        )

        if st.button("🗑️ Delete", key=f"delete_{song['id']}"):
            st.session_state['songs'] = [s for s in st.session_state['songs'] if s['id'] != song['id']]
            st.rerun()
