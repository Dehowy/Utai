import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { getAudioURL, getDownloadURL } from '../api/client'

interface AudioPlayerProps {
  bufferId: string
  compact?: boolean
}

export function AudioPlayer({ bufferId, compact }: AudioPlayerProps) {
  const { t } = useTranslation()
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [hasPlayed, setHasPlayed] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const seekingRef = useRef(false)

  const audioUrl = getAudioURL(bufferId)

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
  }, [bufferId])

  const togglePlay = () => {
    if (!audioRef.current) return
    if (playing) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
      setHasPlayed(true)
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

  if (compact) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div className="utai-player__controls">
          <audio ref={audioRef} src={audioUrl} preload="none" />
          <button
            onClick={togglePlay}
            className={`utai-player__btn ${playing ? 'utai-player__btn--pause' : 'utai-player__btn--play'}`}
          >
            {playing ? '||' : '>>'}
          </button>
          <a
            href={getDownloadURL(bufferId)}
            download
            className="utai-player__btn utai-player__btn--dl"
          >
            DL
          </a>
        </div>
        {hasPlayed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 3, width: '100%' }}>
            <span className="utai-player__time">{formatTime(currentTime)}</span>
            <div className="utai-player__bar" onMouseDown={handleDragSeek}>
              <div className="utai-player__bar-fill" style={{ width: `${progress}%` }} />
              <div className="utai-player__bar-thumb" style={{ left: `${progress}%` }} />
            </div>
            <span className="utai-player__time">{formatTime(duration)}</span>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="utai-player">
      <audio ref={audioRef} src={audioUrl} preload="none" />
      <div className="utai-player__controls" style={{ marginBottom: 4 }}>
        <button
          onClick={togglePlay}
          className={`utai-player__btn ${playing ? 'utai-player__btn--pause' : 'utai-player__btn--play'}`}
        >
          {playing ? '|| ' : '>> '}{t('player.play')}
        </button>
        <a
          href={getDownloadURL(bufferId)}
          download
          className="utai-player__btn utai-player__btn--dl"
        >
          {t('player.download')}
        </a>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <span className="utai-player__time" style={{ minWidth: 32, textAlign: 'right' }}>
          {formatTime(currentTime)}
        </span>
        <div className="utai-player__bar" onMouseDown={handleDragSeek} style={{ minWidth: 120 }}>
          <div className="utai-player__bar-fill" style={{ width: `${progress}%` }} />
          <div className="utai-player__bar-thumb" style={{ left: `${progress}%` }} />
        </div>
        <span className="utai-player__time" style={{ minWidth: 32 }}>
          {formatTime(duration)}
        </span>
      </div>
    </div>
  )
}
