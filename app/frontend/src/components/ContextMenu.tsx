import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { getModelCategories } from '../api/client'
import { onModelsChanged } from '../utils/modelEvents'
import { useExecutionStore } from '../store/executionStore'

interface StaticNodeItem {
  type: string
  labelKey: string
  color: string
}

const STATIC_NODES: StaticNodeItem[] = [
  { type: 'audio_input', labelKey: 'node.audioInput', color: 'var(--accent-green)' },
  { type: 'rvc_conversion', labelKey: 'node.rvcConversion', color: 'var(--accent-pink)' },
  { type: 'media_output', labelKey: 'node.mediaOutput', color: 'var(--accent-orange)' },
]

interface AddableNode {
  type: string
  label: string
  color: string
  /** For separation nodes */
  category?: string
}

interface ContextMenuProps {
  x: number
  y: number
  nodeId: string | null
  edgeId: string | null
  edgeSourceId?: string | null
  edgeTargetId?: string | null
  isRunning?: boolean
  onClose: () => void
  onDeleteNode: (id: string) => void
  onDeleteEdge: (id: string) => void
  onRerunFromNode: (id: string) => void
  onRerunSingleNode?: (id: string) => void
  onAddNode: (type: string, screenX: number, screenY: number, category?: string) => void
}

export function ContextMenu({ x, y, nodeId, edgeId, edgeSourceId, edgeTargetId, isRunning, onClose, onDeleteNode, onDeleteEdge, onRerunFromNode, onRerunSingleNode, onAddNode }: ContextMenuProps) {
  const { t, i18n } = useTranslation()
  // Check if this specific node is in the executing chain (running or pending)
  const nodeStates = useExecutionStore((s) => s.nodeStates)
  const nodeState = nodeId ? nodeStates[nodeId] : undefined
  const isNodeBusy = isRunning && !!nodeState && (nodeState.status === 'running' || nodeState.status === 'pending')
  // Edge is busy if either endpoint is running/pending
  const srcState = edgeSourceId ? nodeStates[edgeSourceId] : undefined
  const tgtState = edgeTargetId ? nodeStates[edgeTargetId] : undefined
  const isEdgeBusy = isRunning && (
    (!!srcState && (srcState.status === 'running' || srcState.status === 'pending')) ||
    (!!tgtState && (tgtState.status === 'running' || tgtState.status === 'pending'))
  )
  const [showAddMenu, setShowAddMenu] = useState(false)
  const [sepItems, setSepItems] = useState<AddableNode[]>([])

  const lang = i18n.language as 'zh' | 'ja' | 'en'

  // Fetch separation categories (only show installed ones)
  useEffect(() => {
    const fetch = () => {
      getModelCategories().then((cats) => {
        setSepItems(
          cats
            .filter((c: any) => c.installed_count > 0)
            .map((c: any) => ({
              type: 'separation',
              label: c[`label_${lang}`] || c.label_zh,
              color: c.color,
              category: c.id,
            }))
        )
      }).catch(console.error)
    }
    fetch()
    return onModelsChanged(fetch)
  }, [lang])

  // Build full addable node list: input + separation categories + conversion + output
  const addableNodes: AddableNode[] = [
    { type: 'audio_input', label: t('node.audioInput'), color: 'var(--accent-green)' },
    ...sepItems,
    { type: 'rvc_conversion', label: t('node.rvcConversion'), color: 'var(--accent-pink)' },
    { type: 'audio_merge', label: t('node.audioMerge'), color: 'var(--accent-gold)' },
    { type: 'media_output', label: t('node.mediaOutput'), color: 'var(--accent-orange)' },
  ]

  return (
    <>
      <div
        className="utai-context-backdrop"
        onClick={onClose}
        onContextMenu={(e) => { e.preventDefault(); onClose() }}
      />
      <div className="utai-context-menu" style={{ left: x, top: y }}>
        {nodeId ? (
          <>
            <button
              className="utai-context-menu__item"
              disabled={isNodeBusy}
              style={isNodeBusy ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
              onClick={() => { if (!isNodeBusy) { onRerunFromNode(nodeId); onClose() } }}
            >
              {t('context.runFromHere')}
            </button>
            {onRerunSingleNode && (
              <button
                className="utai-context-menu__item"
                disabled={isNodeBusy}
                style={isNodeBusy ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
                onClick={() => { if (!isNodeBusy) { onRerunSingleNode(nodeId); onClose() } }}
              >
                {t('context.runThisNode')}
              </button>
            )}
            <div className="utai-context-menu__divider" />
            <button
              className="utai-context-menu__item utai-context-menu__item--danger"
              disabled={isNodeBusy}
              style={isNodeBusy ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
              onClick={() => { if (!isNodeBusy) { onDeleteNode(nodeId); onClose() } }}
            >
              {t('context.deleteNode')}
            </button>
          </>
        ) : edgeId ? (
          <>
            <button
              className="utai-context-menu__item utai-context-menu__item--danger"
              disabled={isEdgeBusy}
              style={isEdgeBusy ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
              onClick={() => { if (!isEdgeBusy) { onDeleteEdge(edgeId); onClose() } }}
            >
              {t('context.deleteEdge')}
            </button>
            <div className="utai-context-menu__divider" />
            <AddNodeSubmenu
              t={t}
              nodes={addableNodes}
              showAddMenu={showAddMenu}
              onToggle={() => setShowAddMenu(!showAddMenu)}
              onAdd={(node) => { onAddNode(node.type, x, y, node.category); onClose() }}
            />
          </>
        ) : (
          <AddNodeSubmenu
            t={t}
            nodes={addableNodes}
            showAddMenu={showAddMenu}
            onToggle={() => setShowAddMenu(!showAddMenu)}
            onAdd={(node) => { onAddNode(node.type, x, y, node.category); onClose() }}
          />
        )}
      </div>
    </>
  )
}

function AddNodeSubmenu({ t, nodes, showAddMenu, onToggle, onAdd }: {
  t: (key: string) => string
  nodes: AddableNode[]
  showAddMenu: boolean
  onToggle: () => void
  onAdd: (node: AddableNode) => void
}) {
  return (
    <div>
      <button className="utai-context-menu__sub-toggle" onClick={onToggle}>
        <span>▸ {t('context.addNode')}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, opacity: 0.5 }}>
          {showAddMenu ? '△' : '▽'}
        </span>
      </button>
      {showAddMenu && (
        <div className="utai-context-menu__sub-items">
          {nodes.map((node, i) => (
            <button
              key={`${node.type}-${node.category || i}`}
              className="utai-context-menu__item"
              style={{ color: node.color }}
              onClick={() => onAdd(node)}
            >
              {node.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
