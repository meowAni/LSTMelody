import React, { useRef, useState, useEffect } from 'react';
import './CustomMusicPlayer.css'

function CustomMusicPlayer({ src }) {
  const audioRef = useRef(null);
  const progressRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;
    const updateProgress = () => {
      if (audio.duration) {
        setProgress((audio.currentTime / audio.duration) * 100);
        setCurrentTime(audio.currentTime);
        setDuration(audio.duration);
      }
    };
    audio.addEventListener('timeupdate', updateProgress);
    return () => audio.removeEventListener('timeupdate', updateProgress);
  }, []);

  const formatTime = (sec) => {
    if (isNaN(sec)) return '0:00';
    const minutes = Math.floor(sec / 60);
    const seconds = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
  };

  const togglePlay = () => {
    const audio = audioRef.current;
    if (isPlaying) audio.pause();
    else audio.play();
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e) => {
    const rect = progressRef.current.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const audio = audioRef.current;
    audio.currentTime = percent * audio.duration;
  };

  return (
    <div className="music-player">
      <mdui-button-icon className="play-button" onClick={togglePlay}>
        {isPlaying ? <mdui-icon name="pause--outlined" /> : <mdui-icon name="play_arrow--outlined" />}
      </mdui-button-icon>

      <div className="progress-container">
        <div className="progress-bar" ref={progressRef} onClick={handleSeek}>
          <div className="progress" style={{width: `${progress}%`}}/>
        </div>
        <div className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>

      <audio ref={audioRef} src={src} />
    </div>
  );
}

export default CustomMusicPlayer;