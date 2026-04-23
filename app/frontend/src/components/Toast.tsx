/** Auto-fading toast notification — non-blocking, bottom-right corner. */

import { useState, useEffect, useCallback } from 'react';

interface ToastItem { id: number; message: string; type: 'info' | 'success' | 'error'; }

let _toastId = 0;
let _addToast: ((msg: string, type?: 'info' | 'success' | 'error') => void) | null = null;

export function showToast(message: string, type: 'info' | 'success' | 'error' = 'info') {
  _addToast?.(message, type);
}

export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  _addToast = useCallback((message: string, type: 'info' | 'success' | 'error' = 'info') => {
    const id = ++_toastId;
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3000);
  }, []);

  if (toasts.length === 0) return null;

  const colors = { info: '#5ce0d8', success: '#5dba5d', error: '#ff6b9d' };

  return (
    <div style={{ position: 'fixed', bottom: 50, right: 16, zIndex: 9999, display: 'flex', flexDirection: 'column', gap: 6, pointerEvents: 'none' }}>
      {toasts.map(t => (
        <div key={t.id} style={{
          background: '#1a1a2e', border: `1px solid ${colors[t.type]}`, color: colors[t.type],
          padding: '6px 14px', fontSize: 12, fontFamily: 'var(--font-pixel)',
          borderRadius: 4, boxShadow: `0 0 8px ${colors[t.type]}40`,
          animation: 'utai-toast-in 0.3s ease-out',
        }}>
          {t.message}
        </div>
      ))}
    </div>
  );
}
