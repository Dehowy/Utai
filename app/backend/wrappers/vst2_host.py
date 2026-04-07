"""Lightweight VST2 host using ctypes — loads a .dll plugin, opens its editor
window with real-time audio output.  GUI interactions (clicking drum pads, etc.)
produce audible sound because editor and audio share the same plugin instance.
"""

import ctypes
import ctypes.wintypes as wt
import threading
import time
import sys
import os
from pathlib import Path

import numpy as np

# intptr_t
c_intptr_t = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32

# ── VST2 constants ──

effOpen = 0; effClose = 1; effSetSampleRate = 10; effSetBlockSize = 11
effMainsChanged = 12; effEditGetRect = 13; effEditOpen = 14; effEditClose = 15
effEditIdle = 19; effGetChunk = 23; effSetChunk = 24; effStartProcess = 71; effStopProcess = 72

audioMasterVersion = 1; audioMasterGetTime = 7; audioMasterGetSampleRate = 16
audioMasterGetBlockSize = 17; audioMasterAutomate = 0; audioMasterIdle = 3

SAMPLE_RATE = 44100
BLOCK_SIZE = 512

# ── AEffect struct (use c_void_p for function pointers) ──

class AEffect(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_int32),
        ("dispatcher", ctypes.c_void_p),
        ("process_deprecated", ctypes.c_void_p),
        ("setParameter", ctypes.c_void_p),
        ("getParameter", ctypes.c_void_p),
        ("numPrograms", ctypes.c_int32),
        ("numParams", ctypes.c_int32),
        ("numInputs", ctypes.c_int32),
        ("numOutputs", ctypes.c_int32),
        ("flags", ctypes.c_int32),
        ("resvd1", ctypes.c_void_p),
        ("resvd2", ctypes.c_void_p),
        ("initialDelay", ctypes.c_int32),
        ("_pad1", ctypes.c_int32),
        ("_pad2", ctypes.c_int32),
        ("ioRatio", ctypes.c_float),
        ("object", ctypes.c_void_p),
        ("user", ctypes.c_void_p),
        ("uniqueID", ctypes.c_int32),
        ("version", ctypes.c_int32),
        ("processReplacing", ctypes.c_void_p),
        ("processDoubleReplacing", ctypes.c_void_p),
        ("future", ctypes.c_char * 56),
    ]

# Callback / function types
HOST_CALLBACK = ctypes.CFUNCTYPE(c_intptr_t, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, c_intptr_t, ctypes.c_void_p, ctypes.c_float)
DISPATCHER_T = ctypes.CFUNCTYPE(c_intptr_t, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, c_intptr_t, ctypes.c_void_p, ctypes.c_float)
PROCESS_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int32)

# VstTimeInfo
class VstTimeInfo(ctypes.Structure):
    _fields_ = [
        ("samplePos", ctypes.c_double), ("sampleRate", ctypes.c_double),
        ("nanoSeconds", ctypes.c_double), ("ppqPos", ctypes.c_double),
        ("tempo", ctypes.c_double), ("barStartPos", ctypes.c_double),
        ("cycleStartPos", ctypes.c_double), ("cycleEndPos", ctypes.c_double),
        ("timeSigNumerator", ctypes.c_int32), ("timeSigDenominator", ctypes.c_int32),
        ("smpteOffset", ctypes.c_int32), ("smpteFrameRate", ctypes.c_int32),
        ("samplesToNextClock", ctypes.c_int32), ("flags", ctypes.c_int32),
    ]


class VST2Host:
    def __init__(self, dll_path: str):
        self.dll_path = dll_path
        self._sample_pos = 0.0
        self._time_info = VstTimeInfo()
        self._time_info.sampleRate = SAMPLE_RATE
        self._time_info.tempo = 120.0
        self._time_info.timeSigNumerator = 4
        self._time_info.timeSigDenominator = 4
        self._time_info.flags = 1 | 2 | (1 << 10)

        self._callback = HOST_CALLBACK(self._host_cb)
        self._dll = ctypes.cdll.LoadLibrary(dll_path)
        entry = self._dll.VSTPluginMain
        entry.restype = ctypes.POINTER(AEffect)
        entry.argtypes = [HOST_CALLBACK]

        ptr = entry(self._callback)
        if not ptr:
            raise RuntimeError("VSTPluginMain returned NULL")
        self._ptr = ptr
        self.effect = ptr.contents
        if self.effect.magic != 0x56737450:
            raise RuntimeError(f"Bad VST magic: {self.effect.magic:#x}")

        # Cast function pointers
        self._dispatcher = DISPATCHER_T(self.effect.dispatcher)
        self._process = PROCESS_T(self.effect.processReplacing)

        # Init
        self._dispatch(effOpen)
        self._dispatch(effSetSampleRate, opt=float(SAMPLE_RATE))
        self._dispatch(effSetBlockSize, value=BLOCK_SIZE)
        self._dispatch(effMainsChanged, value=1)
        self._dispatch(effStartProcess)

        # Audio buffers
        n_out = max(2, self.effect.numOutputs)
        self._out_arrays = [np.zeros(BLOCK_SIZE, dtype=np.float32) for _ in range(n_out)]
        self._out_ptrs = (ctypes.POINTER(ctypes.c_float) * n_out)(
            *[a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in self._out_arrays]
        )
        n_in = max(1, self.effect.numInputs)
        self._in_arrays = [np.zeros(BLOCK_SIZE, dtype=np.float32) for _ in range(n_in)]
        self._in_ptrs = (ctypes.POINTER(ctypes.c_float) * n_in)(
            *[a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in self._in_arrays]
        )

    def _dispatch(self, opcode, index=0, value=0, ptr=None, opt=0.0):
        return self._dispatcher(self._ptr, opcode, index, c_intptr_t(int(value)), ptr, ctypes.c_float(opt))

    def _host_cb(self, effect, opcode, index, value, ptr, opt):
        if opcode == audioMasterVersion: return 2400
        if opcode == audioMasterGetSampleRate: return SAMPLE_RATE
        if opcode == audioMasterGetBlockSize: return BLOCK_SIZE
        if opcode == audioMasterGetTime:
            self._time_info.samplePos = self._sample_pos
            return ctypes.addressof(self._time_info)
        return 0

    def process_block(self) -> np.ndarray:
        for a in self._out_arrays:
            a.fill(0)
        self._process(self._ptr, self._in_ptrs, self._out_ptrs, BLOCK_SIZE)
        self._sample_pos += BLOCK_SIZE
        # Mix all output channels to stereo
        n = len(self._out_arrays)
        if n >= 2:
            # Standard stereo pair from first two channels, plus sum any others
            L = self._out_arrays[0].copy()
            R = self._out_arrays[1].copy()
            for i in range(2, n):
                if np.any(self._out_arrays[i]):
                    L += self._out_arrays[i]
                    R += self._out_arrays[i]
        else:
            L = self._out_arrays[0].copy()
            R = L.copy()
        return np.stack([L, R])

    _STATE_MAGIC = b'UTV2'  # marks state as saved by our VST2 host

    def get_chunk(self) -> bytes | None:
        buf_ptr = ctypes.c_void_p()
        size = self._dispatch(effGetChunk, index=0, ptr=ctypes.byref(buf_ptr))
        if size > 0 and buf_ptr.value:
            return self._STATE_MAGIC + ctypes.string_at(buf_ptr.value, size)
        return None

    def set_chunk(self, data: bytes):
        if not data.startswith(self._STATE_MAGIC):
            return  # ignore incompatible state (e.g. from dawdreamer)
        raw = data[len(self._STATE_MAGIC):]
        buf = ctypes.create_string_buffer(raw)
        self._dispatch(effSetChunk, index=0, value=len(raw), ptr=buf)

    def close(self):
        self._dispatch(effStopProcess)
        self._dispatch(effMainsChanged, value=0)
        self._dispatch(effClose)


def run_editor_with_audio(dll_path: str, state_path: str, owner_hwnd: int = 0):
    """Open VST2 editor with real-time audio. Blocks until window is closed."""
    import sounddevice as sd

    host = VST2Host(dll_path)

    # Load state
    if os.path.exists(state_path):
        try:
            host.set_chunk(open(state_path, "rb").read())
        except Exception:
            pass

    user32 = ctypes.windll.user32

    # Set argtypes for Win32 functions we call
    user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.DefWindowProcW.restype = ctypes.c_long
    user32.DestroyWindow.argtypes = [wt.HWND]
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.SetTimer.argtypes = [wt.HWND, ctypes.c_uint, wt.UINT, ctypes.c_void_p]
    user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
    user32.UpdateWindow.argtypes = [wt.HWND]
    user32.AdjustWindowRect.argtypes = [ctypes.POINTER(wt.RECT), wt.DWORD, wt.BOOL]
    user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
    user32.RegisterClassExW.argtypes = [ctypes.c_void_p]

    # Window proc
    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
    def wnd_proc(hwnd, msg, wp, lp):
        if msg == 0x0010: user32.DestroyWindow(hwnd); return 0       # WM_CLOSE
        if msg == 0x0002: user32.PostQuitMessage(0); return 0         # WM_DESTROY
        if msg == 0x0113: host._dispatch(effEditIdle); return 0       # WM_TIMER
        return user32.DefWindowProcW(hwnd, msg, wp, lp)
    wnd_proc_ref = WNDPROC(wnd_proc)

    class WNDCLASSEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wt.UINT), ("style", wt.UINT), ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
            ("hInstance", wt.HINSTANCE), ("hIcon", wt.HICON), ("hCursor", wt.HANDLE),
            ("hbrBackground", wt.HBRUSH), ("lpszMenuName", wt.LPCWSTR),
            ("lpszClassName", wt.LPCWSTR), ("hIconSm", wt.HICON),
        ]

    wc = WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
    wc.lpfnWndProc = wnd_proc_ref
    wc.hInstance = ctypes.windll.kernel32.GetModuleHandleW(None)
    wc.hbrBackground = ctypes.cast(6, wt.HBRUSH)
    wc.lpszClassName = "UtaiVST"
    user32.RegisterClassExW(ctypes.byref(wc))

    # Editor size
    rect_ptr = ctypes.c_void_p()
    host._dispatch(effEditGetRect, ptr=ctypes.byref(rect_ptr))
    if rect_ptr.value:
        r = ctypes.cast(rect_ptr.value, ctypes.POINTER(ctypes.c_int16 * 4)).contents
        ew, eh = int(r[3] - r[1]), int(r[2] - r[0])
    else:
        ew, eh = 800, 600

    style = 0x00C00000 | 0x00080000 | 0x00020000  # CAPTION|SYSMENU|MINIMIZEBOX
    rect = wt.RECT(0, 0, ew, eh)
    user32.AdjustWindowRect(ctypes.byref(rect), style, False)

    ew_total = rect.right - rect.left
    eh_total = rect.bottom - rect.top

    # Use provided owner HWND, or fall back to foreground window
    if not owner_hwnd:
        user32.GetForegroundWindow.restype = wt.HWND
        owner_hwnd = user32.GetForegroundWindow() or 0
    owner_hwnd_val = wt.HWND(owner_hwnd) if owner_hwnd else None

    user32.CreateWindowExW.restype = wt.HWND
    user32.CreateWindowExW.argtypes = [
        wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wt.HWND, wt.HMENU, wt.HINSTANCE, ctypes.c_void_p,
    ]
    hwnd = user32.CreateWindowExW(
        0, "UtaiVST", "Utai — VST Editor", style,
        100, 100, ew_total, eh_total,
        owner_hwnd_val, None, wc.hInstance, None,
    )

    host._dispatch(effEditOpen, ptr=ctypes.c_void_p(hwnd))

    # Set Utai icon on the window (via class, not SendMessage — safer)
    _icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'electron', 'build', 'icon.ico'))
    if os.path.exists(_icon_path):
        user32.LoadImageW.restype = wt.HANDLE
        user32.LoadImageW.argtypes = [wt.HINSTANCE, wt.LPCWSTR, wt.UINT, ctypes.c_int, ctypes.c_int, wt.UINT]
        _ico_sm = user32.LoadImageW(None, _icon_path, 1, 16, 16, 0x0010)
        _ico_bg = user32.LoadImageW(None, _icon_path, 1, 32, 32, 0x0010)
        if _ico_sm and _ico_bg:
            user32.SetClassLongPtrW.restype = ctypes.c_void_p
            user32.SetClassLongPtrW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_void_p]
            user32.SetClassLongPtrW(hwnd, -34, _ico_sm)  # GCL_HICONSM
            user32.SetClassLongPtrW(hwnd, -14, _ico_bg)  # GCL_HICON

    user32.SetTimer(hwnd, 1, 33, None)  # 30fps idle
    user32.ShowWindow(hwnd, 1)
    user32.UpdateWindow(hwnd)
    # Bring to front: flash TOPMOST then remove it (bypasses Windows focus-steal block)
    user32.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.UINT]
    HWND_TOPMOST = wt.HWND(-1)
    HWND_NOTOPMOST = wt.HWND(-2)
    SWP_NOMOVE = 0x0002; SWP_NOSIZE = 0x0001
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)

    # Audio thread
    audio_running = True
    def audio_loop():
        try:
            stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=2, dtype='float32')
            stream.start()
            buf = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)
            while audio_running:
                audio = host.process_block()
                buf[:, 0] = audio[0]
                buf[:, 1] = audio[1]
                stream.write(np.ascontiguousarray(buf))
            stream.stop(); stream.close()
        except Exception as e:
            print(f'{{"audio_error": "{e}"}}', flush=True)

    t = threading.Thread(target=audio_loop, daemon=True)
    t.start()

    print(f'{{"status": "opened", "audio": true, "size": [{ew},{eh}]}}', flush=True)

    # Message loop
    msg = wt.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))

    # Cleanup
    audio_running = False
    t.join(timeout=3)
    host._dispatch(effEditClose)

    try:
        chunk = host.get_chunk()
        if chunk:
            os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
            open(state_path, "wb").write(chunk)
            print('{"status": "closed", "state_saved": true}', flush=True)
        else:
            print('{"status": "closed", "state_saved": false}', flush=True)
    except Exception as e:
        print(f'{{"status": "closed", "error": "{e}"}}', flush=True)

    host.close()
