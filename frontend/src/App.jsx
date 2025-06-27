import { useState, useEffect } from 'react'
import 'mdui/mdui.css';
import MusicComponent from './components/MusicComponent';
import Sidebar from './components/Sidebar';
import 'mdui/components/button.js';
import 'mdui/components/icon.js';
import './styles.css';

function App() {
  const [musicList, setMusicList] = useState([]);
  const [seed, setSeed] = useState(0);
  const [duration, setDuration] = useState(100);
  const [isLoading, setIsLoading] = useState(false);

  const [instrumentDrums, setInstrumentDrums] = useState(0);
  const [instrumentBass, setInstrumentBass] = useState(0);
  const [instrumentChords, setInstrumentChords] = useState(0);
  const [instrumentLead, setInstrumentLead] = useState(0);

  const createMusic = async () => {
    setIsLoading(true);
    const res = await fetch(`http://localhost:8000/create`, { 
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed, duration }),
    });
    const data = await res.json();
    setMusicList((prev) => [...prev, data]);
    setIsLoading(false);
  };

  const deleteMusic = async (id) => {
    setMusicList((prev) => prev.filter((music) => music.id !== id));
    await fetch(`http://localhost:8000/delete/${id}`, { method: "DELETE" })
  };
  
  return (
    <div className="page">
      <Sidebar createMusic={createMusic} seed={seed} setSeed={setSeed} duration={duration} setDuration={setDuration} instrumentDrums={instrumentDrums} setInstrumentDrums={setInstrumentDrums} instrumentBass={instrumentBass} setInstrumentBass={setInstrumentBass} instrumentChords={instrumentChords} setInstrumentChords={setInstrumentChords} instrumentLead={instrumentLead} setInstrumentLead={setInstrumentLead} isLoading={isLoading}/>
      <div className="gallery">
        {musicList.map((music) => (
          <MusicComponent
            key={music.id}
            id={music.id}
            img={music.img}
            mid={music.mid}
            wav={music.wav}
            onDelete={deleteMusic}
          />
        ))}
      </div>
    </div>
  )
}

export default App