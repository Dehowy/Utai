import { useCallback, useRef, useState, useEffect } from 'react'
import { Handle, Position, useEdges, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { TrainNodeShell } from './TrainNodeShell'
import { useTrainingStore } from '../../store/trainingStore'
import { uploadDataset, getDatasetDir, getDatasetAudioURL } from '../../api/client'

/** Inline mini-player with seekable progress bar */
function FilePlayer({ filePath }: { filePath: string }) {
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rafRef = useRef<number>(0)

  const ensureAudio = useCallback(() => {
    if (!audioRef.current) {
      const audio = new Audio(getDatasetAudioURL(filePath))
      audio.onended = () => { setPlaying(false); setProgress(0) }
      audio.onloadedmetadata = () => setDuration(audio.duration)
      audioRef.current = audio
    }
    return audioRef.current
  }, [filePath])

  useEffect(() => {
    const tick = () => {
      const audio = audioRef.current
      if (audio && playing && audio.duration) {
        setProgress(audio.currentTime / audio.duration)
      }
      if (playing) rafRef.current = requestAnimationFrame(tick)
    }
    if (playing) rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing])

  useEffect(() => {
    return () => {
      audioRef.current?.pause()
      cancelAnimationFrame(rafRef.current)
    }
  }, [])

  const toggle = useCallback(() => {
    const audio = ensureAudio()
    if (playing) {
      audio.pause()
      setPlaying(false)
    } else {
      audio.play().catch(() => {})
      setPlaying(true)
    }
  }, [ensureAudio, playing])

  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const audio = ensureAudio()
    if (!audio.duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    audio.currentTime = ratio * audio.duration
    setProgress(ratio)
  }, [ensureAudio])

  const formatTime = (s: number) => {
    if (!s || !isFinite(s)) return '0:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="utai-file-player" onClick={(e) => e.stopPropagation()}>
      <button
        className={`utai-file-player__btn ${playing ? 'utai-file-player__btn--playing' : ''}`}
        onClick={toggle}
      >
        {playing ? '■' : '▶'}
      </button>
      <div className="utai-file-player__bar" onClick={handleSeek}>
        <div className="utai-file-player__fill" style={{ width: `${progress * 100}%` }} />
      </div>
      {duration > 0 && (
        <span className="utai-file-player__time">{formatTime(progress * duration)}</span>
      )}
    </div>
  )
}

export function TrainAudioImportNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useTrainingStore((s) => s.updateNodeData)
  const setNodeState = useTrainingStore((s) => s.setNodeState)
  const addLog = useTrainingStore((s) => s.addLog)
  const setLogPanelOpen = useTrainingStore((s) => s.setLogPanelOpen)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const params = (data as any)?.params || {}
  const files: string[] = params.files || []
  const datasetDir: string = params.dataset_dir || ''

  const allEdges = useEdges()
  const outgoingEdges = allEdges.filter((e) => e.source === id)

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files
    if (!selectedFiles || selectedFiles.length === 0) return

    const storageKey = `import_${id.replace(/\W/g, '_')}`

    setNodeState(id, { status: 'running', message: t('train.uploading') })
    setLogPanelOpen(true)

    try {
      const fileList = Array.from(selectedFiles)
      addLog(`[import] Uploading ${fileList.length} files...`)

      await uploadDataset(fileList, storageKey)
      const dirRes = await getDatasetDir(storageKey)

      updateNodeData(id, {
        params: {
          ...params,
          files: dirRes.files,
          dataset_dir: dirRes.dataset_dir,
        },
      })

      setNodeState(id, {
        status: 'completed',
        message: `${dirRes.count} ${t('train.filesLoaded')}`,
        outputData: { dir: dirRes.dataset_dir, files: dirRes.files },
      })
      addLog(`[import] Done: ${dirRes.count} files`)
    } catch (err) {
      setNodeState(id, { status: 'error', message: String(err) })
      addLog(`[import] Error: ${err}`)
    }

    e.target.value = ''
  }, [id, params])

  const filePaths = files.map((f) => (datasetDir ? `${datasetDir}/${f}`.replace(/\\/g, '/') : f))

  const connectedFileHandles = new Set(
    outgoingEdges
      .filter((e) => e.sourceHandle?.startsWith('file_'))
      .map((e) => e.sourceHandle)
  )

  const truncName = (name: string, max = 16) => {
    if (name.length <= max) return name
    const ext = name.lastIndexOf('.')
    if (ext > 0 && name.length - ext <= 5) {
      const extStr = name.slice(ext)
      const keep = max - extStr.length - 2
      return keep > 0 ? name.slice(0, keep) + '..' + extStr : name.slice(0, max - 2) + '..'
    }
    return name.slice(0, max - 2) + '..'
  }

  return (
    <TrainNodeShell id={id} title={t('train.node.audioImport')} color="var(--accent-gold)">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="audio/*,.zip"
        style={{ display: 'none' }}
        onChange={handleFileUpload}
      />

      <div
        className="utai-dropzone"
        onClick={() => fileInputRef.current?.click()}
        style={{ minHeight: 36, fontSize: 11 }}
      >
        {files.length > 0
          ? `${files.length} ${t('train.filesLoaded')} — ${t('train.clickToAdd')}`
          : t('train.dropAudio')
        }
      </div>

      {/* Per-file rows: handle RIGHT, name, player */}
      {files.length > 0 && (
        <div className="utai-train-filelist">
          {files.map((f, i) => {
            const handleId = `file_${i}`
            const isConnected = connectedFileHandles.has(handleId)
            return (
              <div
                key={i}
                className="utai-stem-row utai-train-file-row"
                style={{ position: 'relative', opacity: isConnected ? 1 : 0.65 }}
              >
                <span className="utai-train-file-row__name" title={f}>{truncName(f)}</span>
                <FilePlayer filePath={filePaths[i]} />
                <Handle
                  type="source"
                  position={Position.Right}
                  id={handleId}
                  className="utai-handle utai-handle--file"
                  style={{
                    background: isConnected ? 'var(--accent-cyan)' : 'var(--accent-gold)',
                  }}
                />
              </div>
            )
          })}
        </div>
      )}

      {/* Main handle RIGHT — all unrouted files */}
      <div className="utai-stem-row" style={{ position: 'relative', marginTop: 6 }}>
        <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', paddingRight: 6 }}>
          <span className="utai-stem-label" style={{ color: 'var(--accent-gold)', opacity: 0.7, fontSize: 10 }}>
            {t('train.allUnrouted')} ({Math.max(0, files.length - connectedFileHandles.size)})
          </span>
        </div>
        <Handle
          type="source"
          position={Position.Right}
          id="audio_files"
          className="utai-handle"
          style={{ background: 'var(--accent-gold)' }}
        />
      </div>
    </TrainNodeShell>
  )
}
