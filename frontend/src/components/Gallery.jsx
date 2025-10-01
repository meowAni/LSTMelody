import { useState, useEffect } from 'react'
import MusicComponent from './MusicComponent';
import "./Gallery.css";

function Gallery({musicList, deleteMusic}) {
  return (
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
  )
}

export default Gallery;