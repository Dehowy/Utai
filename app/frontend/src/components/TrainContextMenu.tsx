import { useState } from 'react'
import { useTranslation } from 'react-i18next'

interface TrainNodeTypeItem {
  type: string
  labelKey: string
  color: string
}

const TRAIN_NODE_TYPES: TrainNodeTypeItem[] = [
  { type: 'train_audio_import', labelKey: 'train.node.audioImport', color: 'var(--accent-green)' },
  { type: 'train_msst_batch', labelKey: 'train.node.msstBatch', color: 'var(--accent-blue)' },
  { type: 'train_rvc', labelKey: 'train.node.rvcTrain', color: 'var(--accent-pink)' },
]

interface TrainContextMenuProps {
  x: number
  y: number
  nodeId: string | null
  edgeId: string | null
  onClose: () => void
  onDeleteNode: (id: string) => void
  onDeleteEdge: (id: string) => void
  onAddNode: (type: string, screenX: number, screenY: number) => void
}

export function TrainContextMenu({ x, y, nodeId, edgeId, onClose, onDeleteNode, onDeleteEdge, onAddNode }: TrainContextMenuProps) {
  const { t } = useTranslation()
  const [showAddMenu, setShowAddMenu] = useState(false)

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
              className="utai-context-menu__item utai-context-menu__item--danger"
              onClick={() => { onDeleteNode(nodeId); onClose() }}
            >
              {t('context.deleteNode')}
            </button>
            <div className="utai-context-menu__divider" />
            <AddTrainNodeSubmenu
              t={t}
              showAddMenu={showAddMenu}
              onToggle={() => setShowAddMenu(!showAddMenu)}
              onAddNode={(type) => { onAddNode(type, x, y); onClose() }}
            />
          </>
        ) : edgeId ? (
          <>
            <button
              className="utai-context-menu__item utai-context-menu__item--danger"
              onClick={() => { onDeleteEdge(edgeId); onClose() }}
            >
              {t('context.deleteEdge')}
            </button>
            <div className="utai-context-menu__divider" />
            <AddTrainNodeSubmenu
              t={t}
              showAddMenu={showAddMenu}
              onToggle={() => setShowAddMenu(!showAddMenu)}
              onAddNode={(type) => { onAddNode(type, x, y); onClose() }}
            />
          </>
        ) : (
          <AddTrainNodeSubmenu
            t={t}
            showAddMenu={showAddMenu}
            onToggle={() => setShowAddMenu(!showAddMenu)}
            onAddNode={(type) => { onAddNode(type, x, y); onClose() }}
          />
        )}
      </div>
    </>
  )
}

function AddTrainNodeSubmenu({ t, showAddMenu, onToggle, onAddNode }: {
  t: (key: string) => string
  showAddMenu: boolean
  onToggle: () => void
  onAddNode: (type: string) => void
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
          {TRAIN_NODE_TYPES.map((nt) => (
            <button
              key={nt.type}
              className="utai-context-menu__item"
              style={{ color: nt.color }}
              onClick={() => onAddNode(nt.type)}
            >
              {t(nt.labelKey)}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
