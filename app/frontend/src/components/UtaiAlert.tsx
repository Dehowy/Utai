/**
 * UtaiAlert — themed alert/confirm dialog replacing browser native dialogs.
 * Supports two modes:
 *   - alert: single OK button (informational)
 *   - confirm: OK + Cancel buttons
 *
 * Usage via imperative helpers:
 *   import { utaiAlert, utaiConfirm } from './UtaiAlert';
 *   await utaiAlert('Something happened');
 *   const ok = await utaiConfirm('Are you sure?');
 */

import { useState, useEffect, useCallback, createRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';

// ── Imperative API ──────────────────────────────────────────────

let _mountEl: HTMLDivElement | null = null;
let _root: Root | null = null;

function ensureMount() {
  if (!_mountEl) {
    _mountEl = document.createElement('div');
    _mountEl.id = 'utai-alert-root';
    document.body.appendChild(_mountEl);
    _root = createRoot(_mountEl);
  }
}

interface DialogRequest {
  message: string;
  mode: 'alert' | 'confirm';
  resolve: (ok: boolean) => void;
}

let _queue: DialogRequest[] = [];
let _current: DialogRequest | null = null;

function renderCurrent() {
  ensureMount();
  if (!_root) return;

  if (!_current && _queue.length > 0) {
    _current = _queue.shift()!;
  }

  if (_current) {
    const req = _current;
    _root.render(
      <UtaiAlertDialog
        message={req.message}
        mode={req.mode}
        onClose={(ok) => {
          req.resolve(ok);
          _current = null;
          renderCurrent();
        }}
      />
    );
  } else {
    _root.render(null);
  }
}

export function utaiAlert(message: string): Promise<void> {
  return new Promise((resolve) => {
    _queue.push({ message, mode: 'alert', resolve: () => resolve() });
    if (!_current) renderCurrent();
  });
}

export function utaiConfirm(message: string): Promise<boolean> {
  return new Promise((resolve) => {
    _queue.push({ message, mode: 'confirm', resolve });
    if (!_current) renderCurrent();
  });
}

// ── Dialog Component ────────────────────────────────────────────

interface UtaiAlertDialogProps {
  message: string;
  mode: 'alert' | 'confirm';
  onClose: (ok: boolean) => void;
}

function UtaiAlertDialog({ message, mode, onClose }: UtaiAlertDialogProps) {
  const handleOk = useCallback(() => onClose(true), [onClose]);
  const handleCancel = useCallback(() => onClose(false), [onClose]);

  // Escape key closes (cancel for confirm, ok for alert)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        mode === 'confirm' ? handleCancel() : handleOk();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        handleOk();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [mode, handleOk, handleCancel]);

  return (
    <>
      <div
        className="utai-modal-backdrop"
        onClick={mode === 'confirm' ? handleCancel : handleOk}
        style={{ zIndex: 9000 }}
      />
      <div className="utai-confirm" style={{ zIndex: 9001 }}>
        <div className="utai-confirm__msg">{message}</div>
        <div className="utai-confirm__actions">
          {mode === 'confirm' && (
            <button
              className="utai-btn utai-confirm__btn utai-confirm__btn--cancel"
              onClick={handleCancel}
            >
              Cancel
            </button>
          )}
          <button
            className="utai-btn utai-confirm__btn utai-confirm__btn--ok"
            onClick={handleOk}
            autoFocus
          >
            OK
          </button>
        </div>
      </div>
    </>
  );
}
