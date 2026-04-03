import { useCallback, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { AudioPlayer } from '../components/AudioPlayer'
import { getVideoURL, getDownloadURL } from '../api/client'
import { UtaiSelect } from '../components/UtaiSelect'

/* Collapsible sub-section */
function Section({ label, children, defaultOpen }: { label: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen ?? false)
  return (
    <div style={{ marginTop: 3 }}>
      <button
        className="nodrag"
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 3,
          padding: '2px 6px',
          fontSize: 9,
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-pixel)',
          cursor: 'pointer',
          textAlign: 'left',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        {label}
        <span style={{ fontSize: 8 }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: '4px 0 2px' }}>
          {children}
        </div>
      )}
    </div>
  )
}

/* Small labeled field */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="utai-node__field">
      <label className="utai-label">{label}</label>
      {children}
    </div>
  )
}

export function MediaOutputNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const params = (data as any)?.params || {}

  const setParam = useCallback((key: string, value: any) => {
    updateNodeData(id, { params: { ...params, [key]: value } })
  }, [id, params, updateNodeData])

  const outputAudioBuf = nodeState?.outputs?.audio
  const outputVideoPath = nodeState?.outputs?.video_path
  const outputVideoName = nodeState?.outputs?.video_name || outputVideoPath?.split(/[/\\]/).pop() || ''

  // ─── Container format ───
  const container = params.container || 'mp4'
  // ─── Mode: auto-detect from inputs ───
  // mode shown is cosmetic — backend decides based on what inputs arrive

  return (
    <NodeShell id={id} title={t('node.mediaOutput')} color="var(--accent-orange)">
      {/* ── Input handles ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 6 }}>
        <div className="utai-stem-row" style={{ position: 'relative', height: 16, display: 'flex', alignItems: 'center', paddingLeft: 6 }}>
          <Handle
            type="target"
            position={Position.Left}
            id="audio"
            className="utai-handle"
            style={{ background: 'var(--accent-green)' }}
          />
          <span style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--accent-green)', marginLeft: 12 }}>Audio ♪</span>
        </div>
        <div className="utai-stem-row" style={{ position: 'relative', height: 16, display: 'flex', alignItems: 'center', paddingLeft: 6 }}>
          <Handle
            type="target"
            position={Position.Left}
            id="video_path"
            className="utai-handle"
            style={{ background: 'var(--accent-orange)' }}
          />
          <span style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)', marginLeft: 12 }}>Video ▶</span>
        </div>
      </div>

      {/* ═══ Container / Format ═══ */}
      <Field label={t('media.container')}>
        <UtaiSelect
          className="nodrag"
          value={container}
          onChange={(v) => setParam('container', v)}
          style={{ fontSize: 9 }}
          options={[
            { value: 'mp4', label: 'MP4' },
            { value: 'mkv', label: 'MKV' },
            { value: 'webm', label: 'WebM' },
            { value: 'avi', label: 'AVI' },
            { value: 'mov', label: 'MOV' },
            { value: 'flv', label: 'FLV' },
            { value: 'ts', label: 'MPEG-TS' },
            { value: 'wav', label: 'WAV' },
            { value: 'flac', label: 'FLAC' },
            { value: 'mp3', label: 'MP3' },
            { value: 'ogg', label: 'OGG' },
            { value: 'm4a', label: 'M4A (AAC)' },
            { value: 'opus', label: 'Opus' },
            { value: 'aac', label: 'AAC' },
            { value: 'wma', label: 'WMA' },
          ]}
        />
      </Field>

      {/* ═══ Audio Codec ═══ */}
      <Section label={t('media.sectionAudio')} defaultOpen>
        <Field label={t('media.audioCodec')}>
          <UtaiSelect
            className="nodrag"
            value={params.audio_codec || 'auto'}
            onChange={(v) => setParam('audio_codec', v)}
            style={{ fontSize: 9 }}
            options={[
              { value: 'auto', label: t('node.ytAutomatic') },
              { value: 'copy', label: t('media.copy') },
              { value: 'aac', label: 'AAC' },
              { value: 'libmp3lame', label: 'MP3 (LAME)' },
              { value: 'libopus', label: 'Opus' },
              { value: 'libvorbis', label: 'Vorbis' },
              { value: 'flac', label: 'FLAC' },
              { value: 'pcm_s16le', label: 'PCM 16-bit' },
              { value: 'pcm_s24le', label: 'PCM 24-bit' },
              { value: 'pcm_f32le', label: 'PCM 32-bit float' },
              { value: 'alac', label: 'ALAC' },
              { value: 'ac3', label: 'AC3' },
              { value: 'eac3', label: 'E-AC3' },
            ]}
          />
        </Field>

        <Field label={t('media.audioBitrate')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.audio_bitrate || ''}
            onChange={(e) => setParam('audio_bitrate', e.target.value)}
            placeholder="320k / 192k / 128k"
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>

        <Field label={t('media.sampleRate')}>
          <UtaiSelect
            className="nodrag"
            value={params.sample_rate || 'auto'}
            onChange={(v) => setParam('sample_rate', v)}
            style={{ fontSize: 9 }}
            options={[
              { value: 'auto', label: t('node.ytAutomatic') },
              { value: '8000', label: '8000 Hz' },
              { value: '16000', label: '16000 Hz' },
              { value: '22050', label: '22050 Hz' },
              { value: '44100', label: '44100 Hz' },
              { value: '48000', label: '48000 Hz' },
              { value: '96000', label: '96000 Hz' },
            ]}
          />
        </Field>

        <Field label={t('media.audioChannels')}>
          <UtaiSelect
            className="nodrag"
            value={params.audio_channels || 'auto'}
            onChange={(v) => setParam('audio_channels', v)}
            style={{ fontSize: 9 }}
            options={[
              { value: 'auto', label: t('node.ytAutomatic') },
              { value: '1', label: 'Mono' },
              { value: '2', label: 'Stereo' },
              { value: '6', label: '5.1' },
            ]}
          />
        </Field>

        <Field label={t('media.volume')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.volume || ''}
            onChange={(e) => setParam('volume', e.target.value)}
            placeholder="1.0 / 0.5 / 2.0 / -3dB"
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>
      </Section>

      {/* ═══ Video Codec (only when video container) ═══ */}
      {!['wav', 'flac', 'mp3', 'ogg', 'm4a', 'opus', 'aac', 'wma'].includes(container) && (
        <Section label={t('media.sectionVideo')}>
          <Field label={t('media.videoCodec')}>
            <UtaiSelect
              className="nodrag"
              value={params.video_codec || 'auto'}
              onChange={(v) => setParam('video_codec', v)}
              style={{ fontSize: 9 }}
              options={[
                { value: 'auto', label: t('node.ytAutomatic') },
                { value: 'copy', label: t('media.copy') },
                { value: 'libx264', label: 'H.264 (x264)' },
                { value: 'libx265', label: 'H.265 (x265)' },
                { value: 'libvpx-vp9', label: 'VP9' },
                { value: 'libsvtav1', label: 'AV1 (SVT)' },
                { value: 'libaom-av1', label: 'AV1 (libaom)' },
                { value: 'mpeg4', label: 'MPEG-4' },
                { value: 'h264_nvenc', label: 'H.264 (NVENC)' },
                { value: 'hevc_nvenc', label: 'H.265 (NVENC)' },
              ]}
            />
          </Field>

          <Field label={t('media.videoBitrate')}>
            <input
              className="utai-input nodrag"
              type="text"
              value={params.video_bitrate || ''}
              onChange={(e) => setParam('video_bitrate', e.target.value)}
              placeholder="5M / 2M / 8000k"
              style={{ width: '100%', fontSize: 9 }}
            />
          </Field>

          <Field label={t('media.crf')}>
            <input
              className="utai-input nodrag"
              type="text"
              value={params.crf || ''}
              onChange={(e) => setParam('crf', e.target.value)}
              placeholder="23 (H.264) / 28 (H.265)"
              style={{ width: '100%', fontSize: 9 }}
            />
          </Field>

          <Field label={t('media.preset')}>
            <UtaiSelect
              className="nodrag"
              value={params.preset || 'auto'}
              onChange={(v) => setParam('preset', v)}
              style={{ fontSize: 9 }}
              options={[
                { value: 'auto', label: t('node.ytAutomatic') },
                { value: 'ultrafast', label: 'ultrafast' },
                { value: 'superfast', label: 'superfast' },
                { value: 'veryfast', label: 'veryfast' },
                { value: 'faster', label: 'faster' },
                { value: 'fast', label: 'fast' },
                { value: 'medium', label: 'medium' },
                { value: 'slow', label: 'slow' },
                { value: 'slower', label: 'slower' },
                { value: 'veryslow', label: 'veryslow' },
              ]}
            />
          </Field>

          <Field label={t('media.resolution')}>
            <UtaiSelect
              className="nodrag"
              value={params.resolution || 'auto'}
              onChange={(v) => setParam('resolution', v)}
              style={{ fontSize: 9 }}
              options={[
                { value: 'auto', label: t('node.ytAutomatic') },
                { value: '3840:2160', label: '4K (3840x2160)' },
                { value: '2560:1440', label: '1440p (2560x1440)' },
                { value: '1920:1080', label: '1080p (1920x1080)' },
                { value: '1280:720', label: '720p (1280x720)' },
                { value: '854:480', label: '480p (854x480)' },
                { value: '640:360', label: '360p (640x360)' },
              ]}
            />
          </Field>

          <Field label={t('media.fps')}>
            <UtaiSelect
              className="nodrag"
              value={params.fps || 'auto'}
              onChange={(v) => setParam('fps', v)}
              style={{ fontSize: 9 }}
              options={[
                { value: 'auto', label: t('node.ytAutomatic') },
                { value: '24', label: '24' },
                { value: '25', label: '25' },
                { value: '30', label: '30' },
                { value: '50', label: '50' },
                { value: '60', label: '60' },
              ]}
            />
          </Field>

          <Field label={t('media.pixelFormat')}>
            <UtaiSelect
              className="nodrag"
              value={params.pix_fmt || 'auto'}
              onChange={(v) => setParam('pix_fmt', v)}
              style={{ fontSize: 9 }}
              options={[
                { value: 'auto', label: t('node.ytAutomatic') },
                { value: 'yuv420p', label: 'yuv420p' },
                { value: 'yuv444p', label: 'yuv444p' },
                { value: 'yuv420p10le', label: 'yuv420p10le (10-bit)' },
                { value: 'rgb24', label: 'rgb24' },
              ]}
            />
          </Field>
        </Section>
      )}

      {/* ═══ Filters ═══ */}
      <Section label={t('media.sectionFilters')}>
        <Field label={t('media.audioFilter')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.audio_filter || ''}
            onChange={(e) => setParam('audio_filter', e.target.value)}
            placeholder="loudnorm / highpass=f=200"
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>
        <Field label={t('media.videoFilter')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.video_filter || ''}
            onChange={(e) => setParam('video_filter', e.target.value)}
            placeholder="scale=1920:-1 / crop=640:480"
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>
      </Section>

      {/* ═══ Advanced ═══ */}
      <Section label={t('media.sectionAdvanced')}>
        <Field label={t('media.startTime')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.start_time || ''}
            onChange={(e) => setParam('start_time', e.target.value)}
            placeholder="00:00:30 / 30"
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>
        <Field label={t('media.duration')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.duration || ''}
            onChange={(e) => setParam('duration', e.target.value)}
            placeholder="00:01:00 / 60"
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>
        <Field label={t('media.metadata')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.metadata || ''}
            onChange={(e) => setParam('metadata', e.target.value)}
            placeholder="title=My Song;artist=..."
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>
        <Field label={t('media.extraArgs')}>
          <input
            className="utai-input nodrag"
            type="text"
            value={params.extra_args || ''}
            onChange={(e) => setParam('extra_args', e.target.value)}
            placeholder="-movflags +faststart"
            style={{ width: '100%', fontSize: 9 }}
          />
        </Field>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--text-dim)', cursor: 'pointer' }}>
          <input type="checkbox" checked={params.overwrite !== false} onChange={(e) => setParam('overwrite', e.target.checked)} />
          {t('media.overwrite')}
        </label>
      </Section>

      {/* ═══ Output preview ═══ */}
      {outputAudioBuf && !outputVideoPath && (
        <div style={{ marginTop: 6 }}>
          <AudioPlayer bufferId={outputAudioBuf} />
        </div>
      )}
      {outputVideoPath && (
        <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-body)' }}>
          <div style={{ marginBottom: 4, color: 'var(--accent-orange)' }}>▶ {outputVideoName}</div>
          <div style={{ display: 'flex', gap: 4 }}>
            <a
              href={getVideoURL(outputVideoName)}
              download
              className="utai-btn"
              style={{ fontSize: 9, padding: '2px 8px', textDecoration: 'none' }}
            >
              {t('player.download')}
            </a>
          </div>
          {outputAudioBuf && (
            <div style={{ marginTop: 4 }}>
              <AudioPlayer bufferId={outputAudioBuf} compact />
            </div>
          )}
        </div>
      )}
      {!outputAudioBuf && !outputVideoPath && nodeState?.status !== 'running' && nodeState?.status !== 'pending' && (
        <div style={{
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          textAlign: 'center',
          padding: 8,
        }}>
          {t('status.idle')}
        </div>
      )}
    </NodeShell>
  )
}
