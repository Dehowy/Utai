/** Dialog for importing audio files as new tracks or into existing audio tracks. */

import { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';

interface Props {
  onClose: () => void;
}

export default function AudioImportDialog({ onClose }: Props) {
  const { t } = useTranslation();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError('');
    setUploading(true);
    try {
      // Upload to backend audio buffer
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/audio/upload', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const bufferId = data.buffer_id;

      // Get audio info
      const infoRes = await fetch(`/api/audio/${bufferId}/info`);
      const info = await infoRes.json();
      const durationMs = (info.duration ?? 1) * 1000;

      // Create audio track + wave part
      const store = useSynthStore.getState();
      store.addTrack('audio', file.name.replace(/\.[^.]+$/, ''));
      const trackNo = store.project.tracks.length; // addTrack already pushed
      store.addWavePart(trackNo - 1, {
        name: file.name,
        track_no: trackNo - 1,
        position: 0,
        relative_path: '',
        file_duration_ms: durationMs,
        skip: 0,
        trim: 0,
        fadein: 0,
        fadeout: 0,
        buffer_id: bufferId,
      });

      onClose();
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  };

  return (
    <div className="utai-modal-backdrop" onClick={onClose}>
      <div className="utai-synth-import-dialog" onClick={e => e.stopPropagation()}>
        <div className="utai-synth-import-dialog__title">{t('synth.track.importAudio')}</div>
        <div
          className="utai-synth-import-dialog__drop"
          onDragOver={e => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
        >
          {uploading ? '...' : 'WAV / MP3 / FLAC'}
          <br />
          <small style={{ color: 'var(--text-muted)' }}>Drop or click</small>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg"
          style={{ display: 'none' }}
          onChange={onFileSelect}
        />
        {error && <div style={{ color: 'var(--accent-red)', fontSize: 11, marginTop: 6 }}>{error}</div>}
        <button className="utai-btn" onClick={onClose} style={{ marginTop: 8 }}>
          {t('common.cancel') || 'Cancel'}
        </button>
      </div>
    </div>
  );
}
