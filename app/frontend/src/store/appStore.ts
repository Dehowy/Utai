import { create } from 'zustand';

interface AppState {
  modules: { synth: boolean; audio_tools: boolean } | null;
  defaultPage: string;
  loadModules: () => Promise<void>;
  loadPreferences: () => void;
  setDefaultPage: (page: string) => void;
}

const PREFS_KEY = 'utai_preferences';

export const useAppStore = create<AppState>((set) => ({
  modules: null,
  defaultPage: '/synth',

  loadModules: async () => {
    try {
      const res = await fetch('/api/system/modules');
      const data = await res.json();
      let dp = data.default_page || '/';
      const prefs = localStorage.getItem(PREFS_KEY);
      if (prefs) {
        try {
          const saved = JSON.parse(prefs).defaultPage || '';
          const synthOk = data.synth && saved === '/synth';
          const audioOk = data.audio_tools && (saved === '/' || saved === '/train');
          if (synthOk || audioOk) dp = saved;
        } catch {}
      }
      set({ modules: data, defaultPage: dp });
    } catch {
      set({ modules: { synth: true, audio_tools: true }, defaultPage: '/synth' });
    }
  },

  loadPreferences: () => {
    try {
      const raw = localStorage.getItem(PREFS_KEY);
      if (raw) {
        const p = JSON.parse(raw);
        if (p.defaultPage) set({ defaultPage: p.defaultPage });
      }
    } catch {}
  },

  setDefaultPage: (page) => {
    set({ defaultPage: page });
    try {
      const raw = localStorage.getItem(PREFS_KEY);
      const p = raw ? JSON.parse(raw) : {};
      p.defaultPage = page;
      localStorage.setItem(PREFS_KEY, JSON.stringify(p));
    } catch {}
  },
}));
