import { useCallback, useState } from 'react'
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { AudioPlayer } from '../components/AudioPlayer'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { UtaiSelect } from '../components/UtaiSelect'

/* Collapsible sub-section inside Advanced */
function AdvSection({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ marginTop: 2 }}>
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

export function YtdlpNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const updateNodeInternals = useUpdateNodeInternals()
  const params = (data as any)?.params || {}
  const [showAdvanced, setShowAdvanced] = useState(false)

  const outputAudio = params.output_audio !== false
  const outputVideo = params.output_video !== false

  const setParam = useCallback((key: string, value: any) => {
    const next = { ...params, [key]: value }
    updateNodeData(id, { params: next })
    if (key === 'output_audio' || key === 'output_video') {
      setTimeout(() => updateNodeInternals(id), 0)
    }
  }, [id, params, updateNodeData, updateNodeInternals])

  return (
    <NodeShell id={id} title={t('node.ytdlpDownload')} color="var(--accent-orange)">
      {/* URL input */}
      <div className="utai-node__field">
        <label className="utai-label">URL</label>
        <input
          className="utai-input nodrag"
          type="text"
          value={params.url || ''}
          onChange={(e) => setParam('url', e.target.value)}
          placeholder="https://www.youtube.com/watch?v=..."
          style={{ width: '100%', fontSize: 10 }}
        />
      </div>

      {/* ── Output toggles ── */}
      <div className="utai-node__field" style={{ display: 'flex', gap: 10 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--accent-green)', cursor: 'pointer' }}>
          <input type="checkbox" checked={outputAudio} onChange={(e) => setParam('output_audio', e.target.checked)} />
          {t('node.ytOutputAudio')}
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--accent-orange)', cursor: 'pointer' }}>
          <input type="checkbox" checked={outputVideo} onChange={(e) => setParam('output_video', e.target.checked)} />
          {t('node.ytOutputVideo')}
        </label>
      </div>

      {/* ══ Output handle rows — same pattern as MSSTNode ══ */}
      <div style={{ marginTop: 4, borderTop: '1px dotted var(--border-dim)', paddingTop: 4 }}>
        {outputAudio && (
          <div className="utai-stem-row" style={{ position: 'relative', height: 16, display: 'flex', justifyContent: 'flex-end', alignItems: 'center', paddingRight: 6 }}>
            <span style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>Audio ♪</span>
            <Handle
              type="source"
              position={Position.Right}
              id="audio"
              className="utai-handle"
              style={{ background: 'var(--accent-green)' }}
            />
          </div>
        )}
        {outputVideo && (
          <div className="utai-stem-row" style={{ position: 'relative', height: 16, display: 'flex', justifyContent: 'flex-end', alignItems: 'center', paddingRight: 6 }}>
            <span style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>Video ▶</span>
            <Handle
              type="source"
              position={Position.Right}
              id="video_path"
              className="utai-handle"
              style={{ background: 'var(--accent-orange)' }}
            />
          </div>
        )}
      </div>

      {/* ── Format — single text field for full -f control ── */}
      <div className="utai-node__field">
        <label className="utai-label">{t('node.ytFormat')}</label>
        <input
          className="utai-input nodrag"
          type="text"
          value={params.format || ''}
          onChange={(e) => setParam('format', e.target.value)}
          placeholder="bestaudio+bestvideo/best"
          style={{ width: '100%', fontSize: 9 }}
        />
      </div>

      {/* Format sort */}
      <div className="utai-node__field">
        <label className="utai-label">{t('node.ytFormatSort')}</label>
        <input
          className="utai-input nodrag"
          type="text"
          value={params.format_sort || ''}
          onChange={(e) => setParam('format_sort', e.target.value)}
          placeholder="res:720,codec:h264"
          style={{ width: '100%', fontSize: 9 }}
        />
      </div>

      {/* Audio recode (shown when audio output on) */}
      {outputAudio && (
        <div className="utai-node__field">
          <label className="utai-label">{t('node.ytAudioRecode')}</label>
          <UtaiSelect
            className="nodrag"
            value={params.audio_format || 'wav'}
            onChange={(v) => setParam('audio_format', v)}
            style={{ fontSize: 9 }}
            options={[
              { value: 'wav', label: 'WAV' },
              { value: 'mp3', label: 'MP3' },
              { value: 'flac', label: 'FLAC' },
              { value: 'm4a', label: 'M4A' },
              { value: 'opus', label: 'Opus' },
              { value: 'aac', label: 'AAC' },
            ]}
          />
        </div>
      )}

      {/* Video recode (shown when video output on) */}
      {outputVideo && (
        <div className="utai-node__field">
          <label className="utai-label">{t('node.ytVideoRecode')}</label>
          <UtaiSelect
            className="nodrag"
            value={params.recode_video || ''}
            onChange={(v) => setParam('recode_video', v)}
            style={{ fontSize: 9 }}
            options={[
              { value: '', label: t('node.none') },
              { value: 'mp4', label: 'MP4' },
              { value: 'mkv', label: 'MKV' },
              { value: 'webm', label: 'WebM' },
              { value: 'avi', label: 'AVI' },
            ]}
          />
        </div>
      )}

      {/* Strip audio from video (shown when video output on) */}
      {outputVideo && (
        <label className="utai-node__field" style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--text-dim)', cursor: 'pointer' }}>
          <input type="checkbox" checked={params.strip_video_audio || false} onChange={(e) => setParam('strip_video_audio', e.target.checked)} />
          {t('node.ytStripVideoAudio')}
        </label>
      )}

      {/* ── Advanced toggle ── */}
      <button
        className="utai-btn nodrag"
        onClick={() => setShowAdvanced(!showAdvanced)}
        style={{
          fontSize: 9,
          padding: '3px 8px',
          marginTop: 6,
          width: '100%',
          background: 'rgba(255,255,255,0.04)',
          color: 'var(--accent-cyan)',
          border: `1px solid ${showAdvanced ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.12)'}`,
          borderRadius: 3,
          fontFamily: 'var(--font-pixel)',
          letterSpacing: 1,
          cursor: 'pointer',
        }}
      >
        ⚙ {t('train.advanced')} {showAdvanced ? '▲' : '▼'}
      </button>

      {showAdvanced && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, marginTop: 4 }}>
          <AdvSection label={t('node.ytSectionSelection')}>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytPlaylistItems')}</label>
              <input className="utai-input nodrag" type="text" value={params.playlist_items || ''} onChange={(e) => setParam('playlist_items', e.target.value)} placeholder="1-3,5,7-9" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytMaxDownloads')}</label>
              <input className="utai-input nodrag" type="text" value={params.max_downloads || ''} onChange={(e) => setParam('max_downloads', e.target.value)} placeholder="10" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytMatchFilter')}</label>
              <input className="utai-input nodrag" type="text" value={params.match_filter || ''} onChange={(e) => setParam('match_filter', e.target.value)} placeholder="duration>60 & like_count>100" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytDateRange')}</label>
              <input className="utai-input nodrag" type="text" value={params.dateafter || ''} onChange={(e) => setParam('dateafter', e.target.value)} placeholder="20240101" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--text-dim)', cursor: 'pointer' }}>
              <input type="checkbox" checked={params.no_playlist || false} onChange={(e) => setParam('no_playlist', e.target.checked)} />
              {t('node.ytNoPlaylist')}
            </label>
          </AdvSection>

          <AdvSection label={t('node.ytSectionNetwork')}>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytProxy')}</label>
              <input className="utai-input nodrag" type="text" value={params.proxy || ''} onChange={(e) => setParam('proxy', e.target.value)} placeholder="socks5://127.0.0.1:1080" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytRateLimit')}</label>
              <input className="utai-input nodrag" type="text" value={params.limit_rate || ''} onChange={(e) => setParam('limit_rate', e.target.value)} placeholder="1M, 500K" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytSocketTimeout')}</label>
              <input className="utai-input nodrag" type="text" value={params.socket_timeout || ''} onChange={(e) => setParam('socket_timeout', e.target.value)} placeholder="30" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytSourceAddress')}</label>
              <input className="utai-input nodrag" type="text" value={params.source_address || ''} onChange={(e) => setParam('source_address', e.target.value)} placeholder="0.0.0.0" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--text-dim)', cursor: 'pointer' }}>
                <input type="checkbox" checked={params.force_ipv4 || false} onChange={(e) => setParam('force_ipv4', e.target.checked)} />
                IPv4
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--text-dim)', cursor: 'pointer' }}>
                <input type="checkbox" checked={params.force_ipv6 || false} onChange={(e) => setParam('force_ipv6', e.target.checked)} />
                IPv6
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--text-dim)', cursor: 'pointer' }}>
                <input type="checkbox" checked={params.geo_bypass || false} onChange={(e) => setParam('geo_bypass', e.target.checked)} />
                {t('node.ytGeoBypass')}
              </label>
            </div>
          </AdvSection>

          <AdvSection label={t('node.ytSectionAuth')}>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytCookies')}</label>
              <UtaiSelect
                className="nodrag"
                value={params.cookies_from_browser || ''}
                onChange={(v) => setParam('cookies_from_browser', v)}
                style={{ fontSize: 9 }}
                options={[
                  { value: '', label: t('node.none') },
                  { value: 'chrome', label: 'Chrome' },
                  { value: 'firefox', label: 'Firefox' },
                  { value: 'edge', label: 'Edge' },
                  { value: 'safari', label: 'Safari' },
                  { value: 'opera', label: 'Opera' },
                  { value: 'brave', label: 'Brave' },
                ]}
              />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytCookieFile')}</label>
              <input className="utai-input nodrag" type="text" value={params.cookies_file || ''} onChange={(e) => setParam('cookies_file', e.target.value)} placeholder="cookies.txt" style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytUsername')}</label>
              <input className="utai-input nodrag" type="text" value={params.username || ''} onChange={(e) => setParam('username', e.target.value)} style={{ width: '100%', fontSize: 9 }} />
            </div>
            <div className="utai-node__field">
              <label className="utai-label">{t('node.ytPassword')}</label>
              <input className="utai-input nodrag" type="password" value={params.password || ''} onChange={(e) => setParam('password', e.target.value)} style={{ width: '100%', fontSize: 9 }} />
            </div>
          </AdvSection>

          <AdvSection label={t('node.ytSectionPostProcess')}>
            <div className="utai-node__field">
              <label className="utai-label">SponsorBlock</label>
              <UtaiSelect
                className="nodrag"
                value={params.sponsorblock || ''}
                onChange={(v) => setParam('sponsorblock', v)}
                style={{ fontSize: 9 }}
                options={[
                  { value: '', label: t('node.none') },
                  { value: 'mark', label: t('node.ytSponsorMark') },
                  { value: 'remove', label: t('node.ytSponsorRemove') },
                ]}
              />
            </div>
          </AdvSection>
        </div>
      )}

      {/* ── Output previews ── */}
      {nodeState?.outputs?.audio && outputAudio && (
        <div style={{ marginTop: 6 }}>
          <AudioPlayer bufferId={nodeState.outputs.audio} compact />
        </div>
      )}
      {nodeState?.outputs?.video_path && outputVideo && (
        <div style={{ marginTop: 4, fontSize: 9, color: 'var(--accent-orange)' }}>
          {'\u25B6'} {nodeState.outputs.video_title || nodeState.outputs.video_path.split(/[/\\]/).pop()}
        </div>
      )}
    </NodeShell>
  )
}
