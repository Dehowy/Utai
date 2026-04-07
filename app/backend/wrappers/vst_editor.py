"""Standalone VST editor dispatcher — runs in a subprocess.

Usage: python vst_editor.py <dll_path> <state_path> [owner_hwnd]

Routes to the appropriate host:
  VST2 (.dll)  → ctypes vst2_host.py (Python, real-time audio)
  VST3 (.vst3) → UtaiVSTHost.exe     (JUCE C++, real-time audio)
"""

import sys
import os
import json
import subprocess

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: vst_editor.py <dll_path> <state_path> [owner_hwnd]"}))
        sys.exit(1)

    dll_path = sys.argv[1]
    state_path = sys.argv[2]
    owner_hwnd = sys.argv[3] if len(sys.argv) > 3 else "0"

    if not os.path.exists(dll_path):
        print(json.dumps({"error": f"Plugin not found: {dll_path}"}))
        sys.exit(1)

    ext = os.path.splitext(dll_path)[1].lower()

    if ext == ".dll":
        # VST2: ctypes host (Python)
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from app.backend.wrappers.vst2_host import run_editor_with_audio
            run_editor_with_audio(dll_path, state_path, int(owner_hwnd) if owner_hwnd != "0" else 0)
        except Exception as e:
            print(json.dumps({"error": f"VST2 editor failed: {e}"}), flush=True)
            sys.exit(1)
    else:
        # VST3: JUCE host exe
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        exe = os.path.join(project_root, 'VST', 'host', 'UtaiVSTHost.exe')
        if not os.path.exists(exe):
            print(json.dumps({"error": f"JUCE host not found: {exe}"}), flush=True)
            sys.exit(1)
        try:
            proc = subprocess.run(
                [exe, dll_path, state_path, owner_hwnd],
                capture_output=True, text=True,
                cwd=project_root,
            )
            # Forward JUCE host output
            if proc.stdout:
                for line in proc.stdout.strip().split('\n'):
                    if line.strip():
                        print(line.strip(), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"VST3 editor failed: {e}"}), flush=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
