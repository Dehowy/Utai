/** Effects node — applies audio effects chain in the voice conversion workflow. */

import { useState } from 'react'
import { Handle, Position, useReactFlow } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { RENDER_EFFECTS, type RenderEffectType, createDefaultEffect, type TrackEffect } from '../lib/synth/effects'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { AudioPlayer } from '../components/AudioPlayer'

interface EffectsNodeProps {
  id: string
  data: { params: { effects?: TrackEffect[] } }
}

export function EffectsNode({ id, data }: EffectsNodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore(s => s.updateNodeData)
  const nodeState = useExecutionStore(s => s.nodeStates[id])
  const outputBufferId = nodeState?.outputs?.audio
  const effects: TrackEffect[] = data.params?.effects || []
  const [addOpen, setAddOpen] = useState(false)

  const update = (newEffects: TrackEffect[]) => {
    updateNodeData(id, { params: { ...data.params, effects: newEffects } })
  }

  const addEffect = (type: RenderEffectType) => {
    update([...effects, createDefaultEffect(type)])
    setAddOpen(false)
  }

  const removeEffect = (idx: number) => {
    update(effects.filter((_, i) => i !== idx))
  }

  const setParam = (fxIdx: number, key: string, value: number) => {
    const newFx = effects.map((fx, i) =>
      i === fxIdx ? { ...fx, params: { ...fx.params, [key]: value } } : fx
    )
    update(newFx)
  }

  const toggleEnabled = (fxIdx: number) => {
    const newFx = effects.map((fx, i) =>
      i === fxIdx ? { ...fx, enabled: !fx.enabled } : fx
    )
    update(newFx)
  }

  return (
    <NodeShell id={id} title={t('node.effects')} color="var(--accent-cyan)">
      <Handle type="target" position={Position.Left} id="audio" style={{ background: 'var(--accent-cyan)' }} />
      <Handle type="source" position={Position.Right} id="audio" style={{ background: 'var(--accent-cyan)' }} />

      <div style={{ padding: '4px 0' }}>
        {effects.map((fx, fxIdx) => {
          const def = RENDER_EFFECTS[fx.type as RenderEffectType]
          if (!def) return null
          return (
            <div key={fxIdx} style={{
              border: '1px solid var(--border-dim)',
              padding: '3px 5px',
              marginBottom: 3,
              background: fx.enabled ? 'rgba(92,224,216,0.05)' : 'transparent',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                <span
                  style={{ fontSize: 9, color: fx.enabled ? 'var(--accent-cyan)' : 'var(--text-dim)', cursor: 'pointer' }}
                  onClick={() => toggleEnabled(fxIdx)}
                >
                  {fx.enabled ? '\u25cf' : '\u25cb'} {def.label}
                </span>
                <button
                  style={{ background: 'none', border: 'none', color: 'var(--accent-pink)', fontSize: 9, cursor: 'pointer', padding: 0 }}
                  onClick={() => removeEffect(fxIdx)}
                >x</button>
              </div>
              {fx.enabled && Object.entries(def.params).map(([k, p]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ fontSize: 8, color: 'var(--text-dim)', width: 42 }}>{p.label}</span>
                  <input type="range" min={p.min} max={p.max} step={p.step}
                    value={fx.params[k] ?? p.default}
                    onChange={e => setParam(fxIdx, k, parseFloat(e.target.value))}
                    style={{ flex: 1, height: 10 }}
                  />
                  <span style={{ fontSize: 8, color: 'var(--text-secondary)', width: 28, textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {(fx.params[k] ?? p.default).toFixed(p.step < 1 ? 1 : 0)}
                  </span>
                </div>
              ))}
            </div>
          )
        })}

        <div style={{ position: 'relative' }}>
          <button
            className="utai-btn"
            style={{ width: '100%', fontSize: 9, padding: '2px 0', borderColor: 'var(--accent-cyan)', color: 'var(--accent-cyan)' }}
            onClick={() => setAddOpen(!addOpen)}
          >
            + {t('synth.addEffect')}
          </button>
          {addOpen && (
            <div style={{
              position: 'absolute', left: 0, bottom: '100%', zIndex: 10,
              background: 'var(--bg-panel)', border: '1px solid var(--border-mid)',
              width: '100%', padding: 2,
            }}>
              {(Object.entries(RENDER_EFFECTS) as [RenderEffectType, any][]).map(([type, def]) => (
                <button
                  key={type}
                  style={{
                    display: 'block', width: '100%', textAlign: 'left',
                    fontSize: 9, padding: '2px 6px', marginBottom: 1,
                    background: 'transparent', border: '1px solid var(--border-dim)',
                    color: 'var(--text-primary)', cursor: 'pointer',
                  }}
                  onClick={() => addEffect(type)}
                >
                  {def.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {outputBufferId && <AudioPlayer bufferId={outputBufferId} />}
    </NodeShell>
  )
}
