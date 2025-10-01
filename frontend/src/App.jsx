import { useState, useEffect } from 'react'
import 'mdui/mdui.css';
import axios from "axios";
import Sidebar from './components/Sidebar';
import Gallery from './components/Gallery';
import 'mdui/components/button.js';
import 'mdui/components/icon.js';
import './styles.css';

function App() {
  const [musicList, setMusicList] = useState([]);
  const [seed, setSeed] = useState(0);
  const [duration, setDuration] = useState(100);
  const [isLoading, setIsLoading] = useState(false);

  const [instrumentDrums, setInstrumentDrums] = useState(1);
  const [instrumentBass, setInstrumentBass] = useState(1);
  const [instrumentChords, setInstrumentChords] = useState(1);
  const [instrumentLead, setInstrumentLead] = useState(1);

  const createMusic = async () => {
  try {
    if (isLoading) return;
    setIsLoading(true);

    const res = await axios.post(
      "http://localhost:8000/create",
      {
        seed,
        duration,
        instrumentDrums,
        instrumentBass,
        instrumentChords,
        instrumentLead,
      },
      {
        timeout: 120000,
        headers: { "Content-Type": "application/json" },
      }
    );

    setMusicList((prev) => [...prev, res.data]);
    } catch (error) {
      if (axios.isCancel(error)) {
        console.error("Request was canceled:", error.message);
      } else if (error.code === "ECONNABORTED") {
        console.error("Request timed out!");
      } else {
        console.error("Request failed:", error);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const deleteMusic = async (id) => {
    setMusicList((prev) => prev.filter((music) => music.id !== id));
    await fetch(`http://localhost:8000/delete/${id}`, { method: "DELETE" })
  };
  
  return (
    <div className="page">
      <Sidebar createMusic={createMusic} seed={seed} setSeed={setSeed} duration={duration} setDuration={setDuration} instrumentDrums={instrumentDrums} setInstrumentDrums={setInstrumentDrums} instrumentBass={instrumentBass} setInstrumentBass={setInstrumentBass} instrumentChords={instrumentChords} setInstrumentChords={setInstrumentChords} instrumentLead={instrumentLead} setInstrumentLead={setInstrumentLead} isLoading={isLoading}/>
      <Gallery musicList={musicList} deleteMusic={deleteMusic}/>
    </div>
  )
}

export default App