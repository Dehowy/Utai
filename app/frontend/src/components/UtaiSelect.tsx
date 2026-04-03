import { useState, useRef, useEffect, useCallback } from 'react'

interface UtaiSelectProps {
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[] | string[]
  className?: string
  style?: React.CSSProperties
  placeholder?: string
}

export function UtaiSelect({ value, onChange, options, className, style, placeholder }: UtaiSelectProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Normalize options to {value, label} format
  const items = options.map((o) =>
    typeof o === 'string' ? { value: o, label: o } : o
  )

  const matchedItem = items.find((o) => o.value === value)
  const selectedLabel = matchedItem?.label || value || placeholder || ''
  // If value is set but not in options list, it's a "missing" model (e.g. imported project)
  const isMissing = !!value && !matchedItem

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handle = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [open])

  // Close on Escape
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') setOpen(false)
  }, [])

  return (
    <div
      ref={ref}
      className={`utai-custom-select ${className || ''}`}
      style={style}
      onKeyDown={handleKeyDown}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Trigger */}
      <button
        type="button"
        className="utai-custom-select__trigger"
        onClick={() => setOpen(!open)}
      >
        <span className="utai-custom-select__label" style={isMissing ? { color: 'var(--accent-orange)' } : undefined}>
          {isMissing ? `${selectedLabel} \u26A0` : selectedLabel}
        </span>
        <span className="utai-custom-select__arrow">{open ? '\u25B4' : '\u25BE'}</span>
      </button>

      {/* Dropdown */}
      {open && (
        <div className="utai-custom-select__dropdown">
          {items.map((item) => (
            <button
              key={item.value}
              type="button"
              className={`utai-custom-select__option ${item.value === value ? 'utai-custom-select__option--active' : ''}`}
              onClick={() => { onChange(item.value); setOpen(false) }}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
