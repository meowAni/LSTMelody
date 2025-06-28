import React from "react";
import 'mdui/components/slider.js';
import 'mdui/components/icon.js';
import 'mdui/components/text-field.js';
import 'mdui/components/button.js';
import 'mdui/components/divider.js';
import 'mdui/components/circular-progress.js';
import "./Sidebar.css";

function Sidebar({ createMusic, seed, setSeed, duration, setDuration, instrumentDrums, setInstrumentDrums, instrumentBass, setInstrumentBass, instrumentChords, setInstrumentChords, instrumentLead, setInstrumentLead, isLoading }) {
  const maxDuration = 1200;
  const numSeeds = 10;

  const durationSliderRef = React.useRef(null);
  const durationInputRef = React.useRef(null);
  const seedInputRef = React.useRef(null);

  React.useEffect(() => {
    const changeDurationInput = (e) => {
      setDuration(Number(e.target.value));
    }

    const changeSeedInput = (e) => {
      setSeed(e.target.value);
    }

    durationSliderRef.current.labelFormatter = (value) => Number(value);

    durationInputRef.current?.addEventListener('input', changeDurationInput);
    durationSliderRef.current?.addEventListener('input', changeDurationInput);
    seedInputRef.current?.addEventListener('input', changeSeedInput);

    return () => {
      durationInputRef.current?.removeEventListener('input', changeDurationInput);
      durationSliderRef.current?.removeEventListener('input', changeDurationInput);
      seedInputRef.current?.removeEventListener('input', changeSeedInput);
    }
  }, []);
  
  const generateRandomSeed = () => {
    const randSeed = Math.floor(Math.random() * numSeeds);
    setSeed(randSeed);
  };

  return (
    <div className="side-bar">
      <h1 className="header">LstMelody</h1>
      <mdui-divider></mdui-divider>
      <h4><mdui-icon name='access_time--outlined'></mdui-icon>Tokens</h4>
      <div className='duration-box'>
        <mdui-slider className="duration-slider" value={duration} min="0" max={maxDuration} ref={durationSliderRef}></mdui-slider>
        <mdui-text-field className="duration-input" type="number" max={maxDuration} ref={durationInputRef} value={duration} variant="outlined" label="Tokens"></mdui-text-field>
      </div>
      <mdui-divider></mdui-divider>
      <h4><mdui-icon name='spa--outlined'></mdui-icon> Seed</h4>
      <div className='seed-box'>
        <mdui-text-field clearable type="number" min="0" max={numSeeds - 1} value={seed} ref={seedInputRef} variant="outlined" label="Seed"></mdui-text-field>
        <mdui-button className="random-button" variant="tonal" onClick={generateRandomSeed}>
          <mdui-icon name='shuffle--outlined' slot="icon"></mdui-icon>
          Random
        </mdui-button>
      </div>
      <div className='generate-box'>
        {isLoading ? 
        <mdui-button className="generate" loading>Generating...</mdui-button>
        : 
        <mdui-button className="generate" onclick={createMusic}>
          <mdui-icon name='auto_awesome--outlined' slot="icon"></mdui-icon>
          Generate
        </mdui-button>
        }
      </div>
    </div>
  )
}

export default Sidebar;