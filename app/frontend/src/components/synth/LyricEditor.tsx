import { useRef, useEffect, useState } from 'react';
import { useSynthStore } from '../../store/synthStore';

function isCJK(ch: string): boolean {
  const c = ch.codePointAt(0) || 0;
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
         (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0xF900 && c <= 0xFAFF);
}

function isHiragana(ch: string): boolean {
  const c = ch.codePointAt(0) || 0;
  return c >= 0x3040 && c <= 0x309F;
}

function isKatakana(ch: string): boolean {
  const c = ch.codePointAt(0) || 0;
  return (c >= 0x30A0 && c <= 0x30FF) || c === 0x30FC;
}

const REST_LYRICS = new Set(['-', 'R', 'r', 'SP', 'sp']);

export function isValidLyric(lyric: string, language: 'zh' | 'ja'): boolean {
  const s = lyric.trim();
  if (!s) return true;
  if (REST_LYRICS.has(s)) return true;
  if (language === 'zh') return [...s].every(ch => isCJK(ch));
  if (language === 'ja') return [...s].every(ch => isHiragana(ch) || isKatakana(ch));
  return false;
}

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
  const [invalid, setInvalid] = useState(false);
  const mounted = useRef(false);
  const language = useSynthStore(s => s.language);

  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
      mounted.current = true;
    });
    return () => cancelAnimationFrame(raf);
  }, []);

  const tryConfirm = (): boolean => {
    const fallback = language === 'ja' ? 'あ' : '啊';
    const trimmed = text.trim() || fallback;
    if (!isValidLyric(trimmed, language)) {
      setInvalid(true);
      return false;
    }
    setInvalid(false);
    onConfirm(trimmed);
    return true;
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      tryConfirm();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onCancel();
    }
  };

  const handleBlur = () => {
    if (!mounted.current) return;
    // If invalid, just keep editor visible (don't close, don't re-steal focus).
    // Canvas is blocked via lyricEditRef. User can interact with G2P etc. freely.
    tryConfirm();
  };

  return (
    <input
      ref={inputRef}
      value={text}
      onChange={e => { setText(e.target.value); setInvalid(false); }}
      onKeyDown={handleKeyDown}
      onBlur={handleBlur}
      className="utai-synth-lyric-input"
      style={{
        left: x, top: y - 2,
        borderColor: invalid ? 'var(--accent-red)' : undefined,
        boxShadow: invalid ? '0 0 0 1px var(--accent-red)' : undefined,
      }}
    />
  );
}
