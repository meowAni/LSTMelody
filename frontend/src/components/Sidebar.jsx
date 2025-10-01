import React from "react";
import 'mdui/components/slider.js';
import 'mdui/components/icon.js';
import 'mdui/components/text-field.js';
import 'mdui/components/button.js';
import 'mdui/components/divider.js';
import 'mdui/components/circular-progress.js';
import "./Sidebar.css";

function Sidebar({ createMusic, seed, setSeed, duration, setDuration, instrumentDrums, setInstrumentDrums, instrumentBass, setInstrumentBass, instrumentChords, setInstrumentChords, instrumentLead, setInstrumentLead, isLoading }) {
  const minDuration = 100;
  const maxDuration = 2000;
  const numSeeds = 10;

  const durationSliderRef = React.useRef(null);
  const durationInputRef = React.useRef(null);
  const seedInputRef = React.useRef(null);

  const instrumentDrumsIncrement = () => {
    if (instrumentDrums >= 5) return;
    setInstrumentDrums((instrumentDrums) => instrumentDrums + 1);
  };

  const instrumentDrumsDecrement = () => {
    if (instrumentDrums <= 0) return;
    setInstrumentDrums((instrumentDrums) => instrumentDrums - 1);
  }
  
  const instrumentBassIncrement = () => {
    if (instrumentBass >= 5) return;
    setInstrumentBass((instrumentBass) => instrumentBass + 1);
  };

  const instrumentBassDecrement = () => {
    if (instrumentBass <= 0) return;
    setInstrumentBass((instrumentBass) => instrumentBass - 1);
  }
  
  const instrumentChordsIncrement = () => {
    if (instrumentChords >= 5) return;
    setInstrumentChords((instrumentChords) => instrumentChords + 1);
  };

  const instrumentChordsDecrement = () => {
    if (instrumentChords <= 0) return;
    setInstrumentChords((instrumentChords) => instrumentChords - 1);
  }

  const instrumentLeadIncrement = () => {
    if (instrumentLead >= 5) return;
    setInstrumentLead((instrumentLead) => instrumentLead + 1);
  };

  const instrumentLeadDecrement = () => {
    if (instrumentLead <= 0) return;
    setInstrumentLead((instrumentLead) => instrumentLead - 1);
  }

  const resetInstruments = () => {
    setInstrumentDrums(1);
    setInstrumentBass(1);
    setInstrumentChords(1);
    setInstrumentLead(1);
  }

  React.useEffect(() => {
    const changeDurationInput = (e) => {
      let value = Number(e.target.value);
      if (value < minDuration) value = minDuration;
      if (value > maxDuration) value = maxDuration;
      setDuration(value);
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

  const generate = () => {
    if (duration < minDuration) duration = minDuration;
    if (duration > maxDuration) duration = maxDuration;
    createMusic();
  }

  return (
    <div className="side-bar">
      <h1 className="header">LstMelody</h1>
      <mdui-divider></mdui-divider>
      <span className="subheading">
        <mdui-icon name='access_time--outlined'></mdui-icon>
        <span>Tokens</span>
      </span>
      <div>
        <mdui-text-field className="duration-input" type="number" min={minDuration} max={maxDuration} ref={durationInputRef} value={duration} variant="outlined" label="Tokens"></mdui-text-field>
        <mdui-slider className="duration-slider" value={duration} min={minDuration} max={maxDuration} ref={durationSliderRef}></mdui-slider>
      </div>
      <mdui-divider></mdui-divider>
      <span className="subheading">
        <mdui-icon name='spa--outlined'></mdui-icon>
        <span>Seed</span>
      </span>
      <div className='seed-box'>
        <mdui-text-field clearable type="number" min="0" max={numSeeds - 1} value={seed} ref={seedInputRef} variant="outlined" label="Seed"></mdui-text-field>
        <mdui-button className="random-button" variant="tonal" onClick={generateRandomSeed}>
          <mdui-icon name='shuffle--outlined' slot="icon"></mdui-icon>
          Random
        </mdui-button>
      </div>
      <mdui-divider></mdui-divider>
      <span className="subheading">
        <mdui-icon name='piano--outlined'></mdui-icon>
        <span>Instrument Multiplier</span>
      </span>
      <div className='instrument-box'>
        <div className='instrument-stepper'>
          <span className="instrument-name">Drums</span>
          <div className="stepper">
            <mdui-button className="stepper-increment" variant="tonal" onClick={instrumentDrumsIncrement}>+</mdui-button>
            <div className="stepper-data">{instrumentDrums}x</div>
            <mdui-button className="stepper-decrement" variant="tonal" onClick={instrumentDrumsDecrement}>-</mdui-button>
          </div>
        </div>
        <div className='instrument-stepper'>
          <span className="instrument-name">Bass</span>
          <div className="stepper">
            <mdui-button className="stepper-increment" variant="tonal" onClick={instrumentBassIncrement}>+</mdui-button>
            <div className="stepper-data">{instrumentBass}x</div>
            <mdui-button className="stepper-decrement" variant="tonal" onClick={instrumentBassDecrement}>-</mdui-button>
          </div>
        </div>
        <div className='instrument-stepper'>
          <span className="instrument-name">Chords</span>
          <div className="stepper">
            <mdui-button className="stepper-increment" variant="tonal" onClick={instrumentChordsIncrement}>+</mdui-button>
            <div className="stepper-data">{instrumentChords}x</div>
            <mdui-button className="stepper-decrement" variant="tonal" onClick={instrumentChordsDecrement}>-</mdui-button>
          </div>
        </div>
        <div className='instrument-stepper'>
          <span className="instrument-name">Lead</span>
          <div className="stepper">
            <mdui-button className="stepper-increment" variant="tonal" onClick={instrumentLeadIncrement}>+</mdui-button>
            <div className="stepper-data">{instrumentLead}x</div>
            <mdui-button className="stepper-decrement" variant="tonal" onClick={instrumentLeadDecrement}>-</mdui-button>
          </div>
        </div>
      </div>
      <div className="instrument-box-reset"><mdui-button variant="text" onClick={resetInstruments}>Reset</mdui-button></div>
      <mdui-divider className="divider-for-generate"></mdui-divider>
      <div className='generate-box'>
        {isLoading ? 
        <mdui-button className="generate" loading>Generating...</mdui-button>
        : 
        <mdui-button className="generate" onclick={generate}>
          <mdui-icon name='auto_awesome--outlined' slot="icon"></mdui-icon>
          Generate
        </mdui-button>
        }
      </div>
    </div>
  )
}

export default Sidebar;