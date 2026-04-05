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

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
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

  return (
    <input
      ref={inputRef}
      value={text}
      onChange={e => setText(e.target.value)}
      onKeyDown={handleKeyDown}
      onBlur={() => onConfirm(text || 'a')}
      className="utai-synth-lyric-input"
      style={{ left: x, top: y - 2 }}
    />
  );
}
