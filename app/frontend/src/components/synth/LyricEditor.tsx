import { useRef, useEffect, useState } from 'react';

interface Props {
  x: number;
  y: number;
  value: string;
  onConfirm: (value: string) => void;
  onCancel: () => void;
}

export default function LyricEditor({ x, y, value, onConfirm, onCancel }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [text, setText] = useState(value);
  const mounted = useRef(false);

  useEffect(() => {
    // Delay focus so the browser's mousedown focus handling settles first
    const raf = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
      mounted.current = true;
    });
    return () => cancelAnimationFrame(raf);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      onConfirm(text || 'a');
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onCancel();
    }
  };

  const handleBlur = () => {
    // Ignore blur that fires before the input is properly mounted/focused
    if (!mounted.current) return;
    onConfirm(text || 'a');
  };

  return (
    <input
      ref={inputRef}
      value={text}
      onChange={e => setText(e.target.value)}
      onKeyDown={handleKeyDown}
      onBlur={handleBlur}
      className="utai-synth-lyric-input"
      style={{ left: x, top: y - 2 }}
    />
  );
}
