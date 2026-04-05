import { useState, useCallback, useRef, type ReactNode } from 'react';

interface Props {
  left: ReactNode;
  right: ReactNode;
  defaultLeftWidth?: number;
}

export default function SplitPane({ left, right, defaultLeftWidth = 65 }: Props) {
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth);
  const dragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const onMouseDown = useCallback(() => {
    dragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setLeftWidth(Math.max(25, Math.min(80, pct)));
  }, []);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  return (
    <div
      ref={containerRef}
      className="utai-synth-split"
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
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
