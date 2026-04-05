import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { parseUstx, serializeUstx } from '../../lib/synth/ustx';

export default function SynthProjectMenu() {
  const { t } = useTranslation();
  const dirty = useSynthStore(s => s.dirty);
  const project = useSynthStore(s => s.project);
  const loadProject = useSynthStore(s => s.loadProject);
  const newProject = useSynthStore(s => s.newProject);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const proj = parseUstx(reader.result as string);
        loadProject(proj);
      } catch (err) {
        alert(`Failed to load .ustx: ${err}`);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleExport = () => {
    const yaml = serializeUstx(project);
    const blob = new Blob([yaml], { type: 'text/yaml' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${project.name || 'project'}.ustx`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const handleNew = () => {
    if (dirty && !confirm(t('synth.discardChanges'))) return;
    newProject();
  };

  return (
    <>
      <button className="utai-btn" onClick={handleNew} style={{ fontSize: 11 }}>{t('synth.new')}</button>
      <button className="utai-btn" onClick={() => fileInputRef.current?.click()} style={{ fontSize: 11 }}>{t('synth.import')}</button>
      <button className="utai-btn" onClick={handleExport} style={{ fontSize: 11 }}>{t('synth.export')}</button>
      <input ref={fileInputRef} type="file" accept=".ustx,.ust" onChange={handleImport} style={{ display: 'none' }} />
    </>
  );
}
