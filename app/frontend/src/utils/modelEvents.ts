/** Simple event bus for model changes — lets ModelManager notify NodePalette to refresh. */

const EVENT_NAME = 'utai:models-changed'

export function emitModelsChanged() {
  window.dispatchEvent(new CustomEvent(EVENT_NAME))
}

export function onModelsChanged(callback: () => void): () => void {
  window.addEventListener(EVENT_NAME, callback)
  return () => window.removeEventListener(EVENT_NAME, callback)
}
