import { useCallback, useRef, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useWorkflowStore } from '../store/workflowStore'
import { uploadAudio } from '../api/client'

export function AudioInputNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const fileRef = useRef<HTMLInputElement>(null)
  const params = (data as any)?.params || {}
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  const handleFile = useCallback(async (file: File) => {
    setUploading(true)
    setError('')
    try {
      const result = await uploadAudio(file)
      updateNodeData(id, {
        params: {
          buffer_id: result.buffer_id,
          filename: result.filename,
          duration: result.duration,
          sample_rate: result.sample_rate,
        },
      })
    } catch (e: any) {
      console.error('Upload failed:', e)
      setError(e.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [id, updateNodeData])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    fileRef.current?.click()
  }, [])

  return (
    <NodeShell id={id} title={t('node.audioInput')} color="var(--accent-green)">
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); e.stopPropagation() }}
        onClick={handleClick}
        className="nodrag nopan utai-dropzone"
      >
        {uploading ? (
          <span style={{ color: 'var(--accent-gold)', fontFamily: 'var(--font-mono)' }}>uploading...</span>
        ) : params.filename ? (
          <div>
            <div style={{ fontFamily: 'var(--font-pixel)', color: 'var(--accent-green)' }}>
              {params.filename}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3, fontFamily: 'var(--font-mono)' }}>
              {params.duration ? `${params.duration.toFixed(1)}s` : ''}{' '}
              {params.sample_rate ? `${params.sample_rate}Hz` : ''}
            </div>
          </div>
        ) : (
          <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-pixel)', fontSize: 11 }}>
            {t('node.dragHint')}
          </span>
        )}
      </div>
      {error && (
        <div style={{ color: 'var(--accent-red)', fontFamily: 'var(--font-mono)', fontSize: 10, marginTop: 3 }}>
          ! {error}
        </div>
      )}
      <input
        ref={fileRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.wma"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleFile(file)
          e.target.value = ''
        }}
      />
      <Handle type="source" position={Position.Right} id="audio"
        className="utai-handle" style={{ background: 'var(--accent-green)' }} />
    </NodeShell>
  )
}
