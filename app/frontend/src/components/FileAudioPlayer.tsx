import { useEffect, useRef, useState } from 'react'
import { getDatasetAudioURL } from '../api/client'

interface FileAudioPlayerProps {
  filePath: string
  compact?: boolean
}

export function FileAudioPlayer({ filePath, compact = true }: FileAudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const seekingRef = useRef(false)

  const audioUrl = getDatasetAudioURL(filePath)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTimeUpdate = () => { if (!seekingRef.current) setCurrentTime(audio.currentTime) }
    const onDurationChange = () => setDuration(audio.duration || 0)
    const onEnded = () => { setPlaying(false); setCurrentTime(0) }
    audio.addEventListener('timeupdate', onTimeUpdate)
    audio.addEventListener('durationchange', onDurationChange)
    audio.addEventListener('loadedmetadata', onDurationChange)
    audio.addEventListener('ended', onEnded)
    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate)
      audio.removeEventListener('durationchange', onDurationChange)
      audio.removeEventListener('loadedmetadata', onDurationChange)
      audio.removeEventListener('ended', onEnded)
    }
  }, [filePath])

  // Stop when filePath changes
  useEffect(() => {
    setPlaying(false)
    setCurrentTime(0)
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
  }, [filePath])

  const togglePlay = () => {
    if (!audioRef.current) return
    if (playing) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setPlaying(!playing)
  }

  const handleDragSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return
    const bar = e.currentTarget
    seekingRef.current = true
    const seekTo = (clientX: number) => {
      const rect = bar.getBoundingClientRect()
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
      audioRef.current!.currentTime = ratio * duration
      setCurrentTime(ratio * duration)
    }
    seekTo(e.clientX)
    const onMove = (ev: MouseEvent) => seekTo(ev.clientX)
    const onUp = () => {
      seekingRef.current = false
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }

  const formatTime = (s: number) => {
    if (!s || !isFinite(s)) return '0:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0
  const fileName = filePath.split(/[/\\]/).pop() || filePath

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <audio ref={audioRef} src={audioUrl} preload="none" />
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <button
          onClick={togglePlay}
          className={`utai-player__btn ${playing ? 'utai-player__btn--pause' : 'utai-player__btn--play'}`}
          style={{ fontSize: 9, padding: '1px 4px', minWidth: 20 }}
        >
          {playing ? '||' : '\u25B6'}
        </button>
        <span style={{ fontSize: 8, color: 'var(--text-dim)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {fileName}
        </span>
        <span style={{ fontSize: 8, color: 'var(--text-muted)' }}>
          {formatTime(currentTime)}/{formatTime(duration)}
        </span>
      </div>
      {(playing || currentTime > 0) && (
        <div className="utai-player__bar" onMouseDown={handleDragSeek} style={{ height: 4 }}>
          <div className="utai-player__bar-fill" style={{ width: `${progress}%` }} />
          <div className="utai-player__bar-thumb" style={{ left: `${progress}%` }} />
        </div>
      )}
    </div>
  )
}
