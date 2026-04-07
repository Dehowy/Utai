"""Standalone VST editor host — runs in a subprocess with pythonw.exe.

Usage: pythonw.exe vst_editor.py <dll_path> <state_path>

For VST2 (.dll): uses ctypes-based host with real-time audio (click pads = hear sound).
For VST3 (.vst3): uses dawdreamer's open_editor (GUI only, no real-time audio preview).
"""

import sys
import os
import json

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: vst_editor.py <dll_path> <state_path>"}))
        sys.exit(1)

    dll_path = sys.argv[1]
    state_path = sys.argv[2]

    if not os.path.exists(dll_path):
        print(json.dumps({"error": f"DLL not found: {dll_path}"}))
        sys.exit(1)

    ext = os.path.splitext(dll_path)[1].lower()

    if ext == ".dll":
        # VST2: ctypes host with real-time audio
        try:
            # Add project root to path for imports
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from app.backend.wrappers.vst2_host import run_editor_with_audio
            run_editor_with_audio(dll_path, state_path)
        except Exception as e:
            print(json.dumps({"error": f"VST2 editor failed: {e}"}), flush=True)
            sys.exit(1)
    else:
        # VST3: dawdreamer fallback (GUI without real-time audio)
        try:
            import dawdreamer as daw
            engine = daw.RenderEngine(sample_rate=44100, block_size=512)
            plugin = engine.make_plugin_processor("editor", dll_path)
            if os.path.exists(state_path):
                try:
                    plugin.load_state(state_path)
                except Exception:
                    pass
            print(json.dumps({"status": "opening", "name": plugin.get_name(), "audio": False}), flush=True)
            plugin.open_editor()
            try:
                plugin.save_state(state_path)
                print(json.dumps({"status": "closed", "state_saved": True}), flush=True)
            except Exception as e:
                print(json.dumps({"status": "closed", "state_saved": False, "error": str(e)}), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"VST3 editor failed: {e}"}), flush=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
