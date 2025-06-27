import './MusicComponent.css';
import CustomMusicPlayer from './CustomMusicPlayer';

function MusicComponent({ id, img, mid, wav, onDelete }) {
  return (
    <div className='music-box'>
      <img className="preview" src={img} alt="preview" />
      {wav && <CustomMusicPlayer src={wav}/>}
      <div className="button-container">
        <mdui-button href={mid} download="music.midi" className="downloadButton" variant="tonal">
          MIDI
          <mdui-icon slot="icon" name='download--outlined'></mdui-icon>
        </mdui-button>
        <mdui-button href={wav} download="music.wav" target="_blank" className="downloadButton" variant="tonal">
          WAV
          <mdui-icon slot="icon" name='download--outlined'></mdui-icon>
        </mdui-button>
        <mdui-button onClick={() => onDelete(id)} className="deleteButton" variant="tonal">
          Delete
          <mdui-icon slot="icon" name="delete--outlined"></mdui-icon>
        </mdui-button>
      </div>
    </div>
  );
}

export default MusicComponent;
