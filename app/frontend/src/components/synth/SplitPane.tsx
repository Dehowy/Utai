import { useState, useCallback, useRef, useEffect, type ReactNode } from 'react';

interface Props {
  left: ReactNode;
  right: ReactNode;
  defaultLeftWidth?: number;
  rightCollapsed?: boolean;
}

export default function SplitPane({ left, right, defaultLeftWidth = 65, rightCollapsed }: Props) {
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth);
  const dragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const onMouseDown = useCallback(() => {
    dragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setLeftWidth(Math.max(25, Math.min(80, pct)));
  }, []);

  const onMouseUp = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  if (rightCollapsed) {
    return (
      <div ref={containerRef} className="utai-synth-split" style={{ display: 'flex' }}>
        <div style={{ width: '100%', minWidth: 0, overflow: 'hidden', height: '100%' }}>
          {left}
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="utai-synth-split">
      <div style={{ width: `${leftWidth}%`, minWidth: 0, overflow: 'hidden', height: '100%' }}>
        {left}
      </div>
      <div className="utai-synth-split__divider" onMouseDown={onMouseDown} />
      <div style={{ flex: 1, minWidth: 0, overflow: 'hidden', height: '100%' }}>
        {right}
      </div>
    </div>
  );
}
