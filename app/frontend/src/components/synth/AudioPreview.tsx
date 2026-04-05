import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface Props { label: string; audioUrl: string | null; }

export default function AudioPreview({ label, audioUrl }: Props) {
  const { t } = useTranslation();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);

  const toggle = () => {
    if (!audioRef.current || !audioUrl) return;
    if (playing) { audioRef.current.pause(); setPlaying(false); }
    else { audioRef.current.play(); setPlaying(true); }
  };

  const download = () => {
    if (!audioUrl) return;
    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = `${label}.wav`;
    a.click();
  };

  return (
    <div className="utai-synth-audio">
      <div className="utai-synth-audio__title">{label}</div>
      {audioUrl ? (
        <>
          <audio ref={audioRef} src={audioUrl} onEnded={() => setPlaying(false)} />
          <div className="utai-synth-audio__controls">
            <button className="utai-btn" style={{ fontSize: 11, padding: '2px 8px' }} onClick={toggle}>
              {playing ? t('synth.stop') : t('synth.play')}
            </button>
            <button className="utai-btn" style={{ fontSize: 11, padding: '2px 8px' }} onClick={download}>
              {t('synth.download')}
            </button>
          </div>
        </>
      ) : (
        <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{t('synth.noAudio')}</div>
      )}
    </div>
  );
}
