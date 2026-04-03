"""Wrapper for RVC Training Pipeline.

Wraps the multi-step RVC training process:
  1. Preprocess (audio slicing + resampling)
  2. F0 extraction
  3. Feature extraction (HuBERT)
  4. Model training
  5. Index building (FAISS)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from app.backend.config import RVC_ROOT, TEMP_DIR

logger = logging.getLogger(__name__)

# Training experiment base directory
EXPERIMENTS_DIR = RVC_ROOT / "logs"


class TrainingTask:
    """Represents a running training task with cancellation support."""

    def __init__(self, task_id: str, exp_name: str):
        self.task_id = task_id
        self.exp_name = exp_name
        self.current_step = ""
        self.progress = 0.0
        self.status = "idle"  # idle, running, completed, error, cancelled
        self.logs: list[str] = []
        self._process: subprocess.Popen | None = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        if self._process:
            try:
                pid = self._process.pid
                if pid:
                    RVCTrainWrapper._kill_process_tree(pid)
            except (OSError, ProcessLookupError):
                pass


class RVCTrainWrapper:
    """Wraps the RVC training pipeline."""

    def __init__(self):
        self._rvc_root = str(RVC_ROOT)
        self._tasks: dict[str, TrainingTask] = {}

    def get_task(self, task_id: str) -> TrainingTask | None:
        return self._tasks.get(task_id)

    @staticmethod
    def _kill_process_tree(pid: int):
        """Kill a process and all its children (Windows process tree kill)."""
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                )
            except Exception:
                pass
        else:
            try:
                import signal
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass

    def list_experiments(self) -> list[dict[str, Any]]:
        """List existing training experiments."""
        experiments = []
        if not EXPERIMENTS_DIR.exists():
            return experiments
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if not exp_dir.is_dir():
                continue
            # Check if it looks like an experiment
            has_config = (exp_dir / "config.json").exists()
            has_wavs = (exp_dir / "0_gt_wavs").exists()
            has_model = any(exp_dir.glob("G_*.pth"))
            if has_config or has_wavs or has_model:
                # Count training progress
                steps = {}
                if has_wavs:
                    steps["preprocess"] = True
                if (exp_dir / "2a_f0").exists():
                    steps["f0"] = True
                feat_dir = exp_dir / "3_feature768"
                if not feat_dir.exists():
                    feat_dir = exp_dir / "3_feature256"
                if feat_dir.exists():
                    steps["features"] = True
                if has_model:
                    steps["training"] = True
                if any(exp_dir.glob("added_*.index")):
                    steps["index"] = True

                experiments.append({
                    "name": exp_dir.name,
                    "path": str(exp_dir),
                    "completed_steps": steps,
                })

        return experiments

    def list_pretrained_models(self, version: str = "v2") -> dict[str, list[str]]:
        """List available pretrained G and D models for the given RVC version."""
        result = {"G": [], "D": []}
        suffix = f"_{version}" if version != "v1" else ""
        d = Path(self._rvc_root) / "assets" / f"pretrained{suffix}"
        if not d.exists():
            return result
        for f in sorted(d.glob("*.pth")):
            name = f.name
            rel = str(f)
            if name.startswith("f0G") or name.startswith("G"):
                result["G"].append(rel)
            elif name.startswith("f0D") or name.startswith("D"):
                result["D"].append(rel)
        return result

    def _get_python_executable(self) -> str:
        """Get the Python executable path from venv or fallback."""
        # Use the same Python that's running the server
        return sys.executable

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for RVC subprocesses."""
        env = os.environ.copy()
        env["weight_root"] = os.path.join(self._rvc_root, "assets", "weights")
        env["weight_uvr5_root"] = os.path.join(self._rvc_root, "assets", "uvr5_weights")
        env["index_root"] = os.path.join(self._rvc_root, "logs")
        env["outside_index_root"] = os.path.join(self._rvc_root, "assets", "indices")
        env["rmvpe_root"] = os.path.join(self._rvc_root, "assets", "rmvpe")
        env["OPENBLAS_NUM_THREADS"] = "1"
        # Force UTF-8 encoding for subprocess stdout/stderr to handle
        # Japanese/Chinese filenames without UnicodeEncodeError on Windows
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        # Force unbuffered stdout/stderr so training logs stream in real-time
        env["PYTHONUNBUFFERED"] = "1"
        # Note: PYTHONPATH is ignored when python310._pth is present (portable mode).
        # Stubs (tensorboard etc.) are installed directly into site-packages instead.
        # Disable libuv for torch.distributed (not available in all PyTorch builds)
        env["USE_LIBUV"] = "0"
        # Pass RVC version for child process imports (see train.py module-level guard)
        env["RVC_VERSION"] = "v2"
        return env

    async def preprocess(
        self,
        task: TrainingTask,
        exp_name: str,
        input_dir: str,
        sr: int,
        n_processes: int = 4,
        log_callback: Callable | None = None,
    ) -> bool:
        """Step 1: Audio preprocessing (slicing + resampling)."""
        task.current_step = "preprocess"
        task.status = "running"

        exp_dir = str(EXPERIMENTS_DIR / exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        python = self._get_python_executable()
        cmd = [
            python,
            os.path.join(self._rvc_root, "infer", "modules", "train", "preprocess.py"),
            input_dir,
            str(sr),
            str(n_processes),
            exp_dir,
            "False",  # noparallel
            "3.7",  # per-slice duration
        ]

        if log_callback:
            await log_callback(f"[preprocess] Starting: sr={sr}, processes={n_processes}")
            await log_callback(f"[preprocess] Input: {input_dir}")

        success = await self._run_subprocess(task, cmd, log_callback)

        # Verify preprocessing actually produced output files
        if success:
            gt_dir = Path(exp_dir) / "0_gt_wavs"
            wav_count = len(list(gt_dir.glob("*.wav"))) if gt_dir.exists() else 0
            if wav_count == 0:
                if log_callback:
                    await log_callback(
                        "[preprocess] ERROR: No output files produced. "
                        "Subprocess may have silently failed (e.g. encoding errors with unicode filenames)."
                    )
                success = False
            else:
                if log_callback:
                    await log_callback(f"[preprocess] Produced {wav_count} audio slices")

        # On failure, try to read the preprocess.log for more details
        if not success and log_callback:
            log_file = Path(exp_dir) / "preprocess.log"
            if log_file.exists():
                try:
                    content = log_file.read_text(encoding="utf-8", errors="replace").strip()
                    if content:
                        await log_callback("[preprocess] Log file contents:")
                        for line in content.splitlines()[-20:]:
                            await log_callback(f"  {line}")
                except Exception:
                    pass

        return success

    async def extract_f0(
        self,
        task: TrainingTask,
        exp_name: str,
        f0_method: str = "rmvpe",
        n_processes: int = 4,
        log_callback: Callable | None = None,
    ) -> bool:
        """Step 2: F0 (pitch) extraction."""
        task.current_step = "f0_extract"
        task.status = "running"

        exp_dir = str(EXPERIMENTS_DIR / exp_name)

        if f0_method == "rmvpe_gpu":
            script = os.path.join(self._rvc_root, "infer", "modules", "train", "extract", "extract_f0_rmvpe.py")
            cmd = [
                self._get_python_executable(),
                script,
                "1",  # n_part (leng)
                "0",  # i_part (idx)
                "0",  # i_gpu
                exp_dir,
                "True",  # is_half
            ]
        else:
            script = os.path.join(self._rvc_root, "infer", "modules", "train", "extract", "extract_f0_print.py")
            cmd = [
                self._get_python_executable(),
                script,
                exp_dir,
                str(n_processes),
                f0_method,
            ]

        if log_callback:
            await log_callback(f"[f0] Starting F0 extraction: method={f0_method}")

        success = await self._run_subprocess(task, cmd, log_callback)
        return success

    async def extract_features(
        self,
        task: TrainingTask,
        exp_name: str,
        version: str = "v2",
        gpu_id: int = 0,
        log_callback: Callable | None = None,
    ) -> bool:
        """Step 3: Feature extraction (HuBERT)."""
        task.current_step = "feature_extract"
        task.status = "running"

        exp_dir = str(EXPERIMENTS_DIR / exp_name)

        script = os.path.join(self._rvc_root, "infer", "modules", "train", "extract_feature_print.py")
        # Script expects 8 args: device n_part i_part i_gpu exp_dir version is_half
        cmd = [
            self._get_python_executable(),
            script,
            "cuda" if gpu_id >= 0 else "cpu",
            "1",  # n_part
            "0",  # i_part
            str(gpu_id) if gpu_id >= 0 else "0",
            exp_dir,
            version,
            "True",  # is_half (use FP16 on GPU)
        ]

        if log_callback:
            await log_callback(f"[features] Starting feature extraction: version={version}, GPU={gpu_id}")

        success = await self._run_subprocess(task, cmd, log_callback)
        return success

    async def write_filelist(
        self,
        exp_name: str,
        sr: int,
        version: str = "v2",
        log_callback: Callable | None = None,
    ) -> bool:
        """Generate the training filelist (gt_wavs, feature, f0 paths)."""
        exp_dir = EXPERIMENTS_DIR / exp_name

        # Determine feature dimension
        feat_dim = "768" if version == "v2" else "256"
        feat_dir = exp_dir / f"3_feature{feat_dim}"
        gt_dir = exp_dir / "0_gt_wavs"
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"

        if not feat_dir.exists() or not gt_dir.exists():
            if log_callback:
                await log_callback("[filelist] ERROR: Feature or wav directory not found")
            return False

        lines = []
        for feat_file in sorted(feat_dir.glob("*.npy")):
            name = feat_file.stem  # e.g. "0_0"
            # gt_wavs contains .wav files, f0 dirs contain .wav.npy files
            gt_path = gt_dir / f"{name}.wav"
            f0_path = f0_dir / f"{name}.wav.npy"
            f0nsf_path = f0nsf_dir / f"{name}.wav.npy"

            if gt_path.exists():
                # Format: gt_wav_path|feature_path|f0_path|f0nsf_path|speaker_id
                line = f"{gt_path.as_posix()}|{feat_file.as_posix()}|{f0_path.as_posix()}|{f0nsf_path.as_posix()}|0"
                lines.append(line)

        if not lines:
            if log_callback:
                await log_callback("[filelist] ERROR: No matching files found")
            return False

        filelist_path = exp_dir / "filelist.txt"
        filelist_path.write_text("\n".join(lines), encoding="utf-8")

        if log_callback:
            await log_callback(f"[filelist] Generated filelist with {len(lines)} entries")

        # Write config.json
        sr_map = {"32k": 32000, "40k": 40000, "48k": 48000}
        sr_label = f"{sr // 1000}k" if sr >= 1000 else str(sr)
        config_src = Path(self._rvc_root) / "configs" / version / f"{sr_label}.json"

        if config_src.exists():
            config = json.loads(config_src.read_text(encoding="utf-8"))
        else:
            # Fallback to 40k
            config_src = Path(self._rvc_root) / "configs" / version / "40k.json"
            config = json.loads(config_src.read_text(encoding="utf-8"))

        config["data"]["training_files"] = filelist_path.as_posix()
        config_dst = exp_dir / "config.json"
        config_dst.write_text(json.dumps(config, indent=2), encoding="utf-8")

        if log_callback:
            await log_callback(f"[filelist] Config written: {config_dst}")

        return True

    async def train(
        self,
        task: TrainingTask,
        exp_name: str,
        sr: str = "40k",
        f0: bool = True,
        batch_size: int = 4,
        total_epoch: int = 100,
        save_epoch: int = 10,
        gpu_ids: str = "0",
        pretrained_g: str = "",
        pretrained_d: str = "",
        save_latest: bool = True,
        cache_gpu: bool = False,
        save_weights: bool = True,
        version: str = "v2",
        log_callback: Callable | None = None,
    ) -> bool:
        """Step 4: Model training."""
        task.current_step = "training"
        task.status = "running"
        task._total_epoch = total_epoch  # Store for progress calculation

        python = self._get_python_executable()

        # Auto-resolve pretrained models if not explicitly set.
        # RVC needs pretrained G/D as a starting point — training from scratch
        # produces terrible quality ("electric" sounding voice).
        # Format: f0G48k.pth (with f0) or G48k.pth (without f0)
        if not pretrained_g:
            prefix = "f0" if f0 else ""
            pretrained_g = os.path.join(
                self._rvc_root, "assets", f"pretrained_{version}",
                f"{prefix}G{sr}.pth"
            )
            if not os.path.exists(pretrained_g):
                pretrained_g = ""
        if not pretrained_d:
            prefix = "f0" if f0 else ""
            pretrained_d = os.path.join(
                self._rvc_root, "assets", f"pretrained_{version}",
                f"{prefix}D{sr}.pth"
            )
            if not os.path.exists(pretrained_d):
                pretrained_d = ""

        cmd = [
            python,
            os.path.join(self._rvc_root, "infer", "modules", "train", "train.py"),
            "-e", exp_name,
            "-sr", sr,
            "-f0", "1" if f0 else "0",
            "-bs", str(batch_size),
            "-g", gpu_ids,
            "-te", str(total_epoch),
            "-se", str(save_epoch),
            "-l", "1" if save_latest else "0",
            "-c", "1" if cache_gpu else "0",
            "-sw", "1" if save_weights else "0",
            "-v", version,
        ]

        if pretrained_g:
            cmd += ["-pg", pretrained_g]
        if pretrained_d:
            cmd += ["-pd", pretrained_d]

        if log_callback:
            await log_callback(f"[train] Starting training: epochs={total_epoch}, batch={batch_size}")
            await log_callback(f"[train] Config: sr={sr}, f0={f0}, version={version}")
            if pretrained_g:
                await log_callback(f"[train] Pretrained G: {os.path.basename(pretrained_g)}")
            if pretrained_d:
                await log_callback(f"[train] Pretrained D: {os.path.basename(pretrained_d)}")
            if not pretrained_g and not pretrained_d:
                await log_callback("[train] WARNING: No pretrained models found — training from scratch!")

        success = await self._run_subprocess(task, cmd, log_callback, extra_env={"RVC_VERSION": version})
        return success

    async def build_index(
        self,
        task: TrainingTask,
        exp_name: str,
        version: str = "v2",
        log_callback: Callable | None = None,
    ) -> bool:
        """Step 5: Build FAISS index."""
        task.current_step = "index_build"
        task.status = "running"

        if log_callback:
            await log_callback(f"[index] Building FAISS index for {exp_name}...")

        try:
            import faiss
            import numpy as np

            exp_dir = EXPERIMENTS_DIR / exp_name
            feat_dim = "768" if version == "v2" else "256"
            feat_dir = exp_dir / f"3_feature{feat_dim}"

            if not feat_dir.exists():
                if log_callback:
                    await log_callback(f"[index] ERROR: Feature directory not found: {feat_dir}")
                return False

            # Load all features
            npys = []
            for name in sorted(os.listdir(str(feat_dir))):
                if name.endswith(".npy"):
                    phone = np.load(str(feat_dir / name))
                    npys.append(phone)

            if not npys:
                if log_callback:
                    await log_callback("[index] ERROR: No feature files found")
                return False

            big_npy = np.concatenate(npys, 0)
            if log_callback:
                await log_callback(f"[index] Loaded {big_npy.shape[0]} feature vectors ({feat_dim}D)")

            # K-means reduction for large datasets
            n_cpu = os.cpu_count() or 4
            if big_npy.shape[0] > 200000:
                if log_callback:
                    await log_callback("[index] Large dataset, applying K-means reduction...")
                from sklearn.cluster import MiniBatchKMeans
                big_npy = (
                    MiniBatchKMeans(
                        n_clusters=10000,
                        verbose=True,
                        batch_size=256 * n_cpu,
                        compute_labels=False,
                        init="random",
                    )
                    .fit(big_npy)
                    .cluster_centers_
                )

            # Build index
            dim = int(feat_dim)
            n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
            n_ivf = max(n_ivf, 1)

            if log_callback:
                await log_callback(f"[index] Creating IVF{n_ivf} index...")

            index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
            index_ivf = faiss.extract_index_ivf(index)
            index_ivf.nprobe = 1
            index.train(big_npy)

            # Add in batches
            batch_size_idx = 8192
            for i in range(0, big_npy.shape[0], batch_size_idx):
                index.add(big_npy[i : i + batch_size_idx])

            # Save index
            index_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_name}_{version}.index"
            faiss.write_index(index, str(index_path))

            if log_callback:
                await log_callback(f"[index] Index saved: {index_path}")

            return True

        except Exception as e:
            if log_callback:
                await log_callback(f"[index] ERROR: {e}")
            logger.exception("Index building failed")
            return False

    async def export_model(
        self,
        exp_name: str,
        output_dir: str | None = None,
        log_callback: Callable | None = None,
    ) -> str | None:
        """Export the trained model weights to assets/weights/ (and optionally a custom dir)."""
        exp_dir = EXPERIMENTS_DIR / exp_name
        weights_dir = Path(self._rvc_root) / "assets" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Find the latest saved weight file
        weight_files = sorted(exp_dir.glob(f"{exp_name}_*.pth"))
        if not weight_files:
            # Try weights directory itself
            weight_files = sorted(weights_dir.glob(f"{exp_name}*.pth"))

        if not weight_files:
            if log_callback:
                await log_callback(f"[export] No weight files found for {exp_name}")
            return None

        latest = weight_files[-1]
        dest = weights_dir / f"{exp_name}.pth"

        if latest != dest:
            shutil.copy2(str(latest), str(dest))

        if log_callback:
            await log_callback(f"[export] Model exported to {dest}")

        # Copy to custom output dir if specified
        if output_dir:
            custom_dest = Path(output_dir) / f"{exp_name}.pth"
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(str(latest), str(custom_dest))
            if log_callback:
                await log_callback(f"[export] Also copied to {custom_dest}")

        # Copy index file too
        index_files = list(exp_dir.glob("added_*.index"))
        if index_files:
            idx_src = index_files[-1]
            idx_dest = weights_dir.parent / "indices" / idx_src.name
            os.makedirs(idx_dest.parent, exist_ok=True)
            shutil.copy2(str(idx_src), str(idx_dest))
            if log_callback:
                await log_callback(f"[export] Index exported to {idx_dest}")

        return str(dest)

    async def _run_subprocess(
        self,
        task: TrainingTask,
        cmd: list[str],
        log_callback: Callable | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> bool:
        """Run a subprocess and stream its output."""
        env = self._build_env()
        if extra_env:
            env.update(extra_env)

        # On Windows: CREATE_NO_WINDOW prevents console popups,
        # CREATE_NEW_PROCESS_GROUP enables killing the entire process tree
        kwargs: dict = {}
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._rvc_root,
                env=env,
                **kwargs,
            )
            task._process = process

            stderr_lines: list[str] = []
            _seen_lines: set[str] = set()  # dedup between pipe and logfile

            def _process_line(text: str, is_stderr: bool = False):
                """Process a log line (dedup, store, forward, parse progress)."""
                if text in _seen_lines:
                    return
                _seen_lines.add(text)
                if is_stderr:
                    stderr_lines.append(text)
                task.logs.append(text)

            async def _forward_line(text: str):
                """Forward to callback and parse epoch progress."""
                if log_callback:
                    await log_callback(text)
                epoch_match = re.search(r"(?:Train )?Epoch:\s*(\d+)", text)
                if epoch_match:
                    current = int(epoch_match.group(1))
                    total = getattr(task, '_total_epoch', 0)
                    if total > 0:
                        task.progress = min(current / total, 1.0)

            async def _read_stream(stream, is_stderr=False):
                """Read a stream line by line, forwarding to logs."""
                async for line in stream:
                    if task._cancelled:
                        return
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if not text:
                        continue
                    _process_line(text, is_stderr)
                    await _forward_line(text)

            async def _tail_logfile():
                """Tail train.log as backup for mp.Process workers whose
                stdout doesn't connect to our pipe (multi-GPU scenario)."""
                # Detect the experiment log file from the command args
                log_path = None
                for i, arg in enumerate(cmd):
                    if arg == "-e" and i + 1 < len(cmd):
                        exp_name = cmd[i + 1]
                        log_path = Path(self._rvc_root) / "logs" / exp_name / "train.log"
                        break
                if not log_path:
                    return

                # Wait for the file to appear
                for _ in range(60):
                    if task._cancelled or process.returncode is not None:
                        return
                    if log_path.exists():
                        break
                    await asyncio.sleep(0.5)
                else:
                    return

                # Tail the file
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        while not task._cancelled and process.returncode is None:
                            line = f.readline()
                            if line:
                                text = line.rstrip()
                                if text and text not in _seen_lines:
                                    _process_line(text)
                                    await _forward_line(text)
                            else:
                                await asyncio.sleep(0.3)
                except Exception:
                    pass

            # Read stdout, stderr, AND tail logfile in parallel
            stdout_task = asyncio.create_task(_read_stream(process.stdout, is_stderr=False))
            stderr_task = asyncio.create_task(_read_stream(process.stderr, is_stderr=True))
            logfile_task = asyncio.create_task(_tail_logfile())

            # Wait for process to finish (stdout/stderr EOF signals this)
            await asyncio.gather(stdout_task, stderr_task)
            await process.wait()

            # Give logfile tailer a moment to catch final lines, then cancel
            await asyncio.sleep(0.5)
            logfile_task.cancel()
            try:
                await logfile_task
            except asyncio.CancelledError:
                pass

            if task._cancelled:
                self._kill_process_tree(process.pid)
                if log_callback:
                    await log_callback("[cancelled] Task was cancelled by user")
                task.status = "cancelled"
                return False

            # RVC uses os._exit(2333333) as a SUCCESS signal after training completes
            if process.returncode not in (0, 2333333):
                if log_callback:
                    await log_callback(f"[error] Process exited with code {process.returncode}")
                    # Log stderr output (contains Python tracebacks)
                    if stderr_lines:
                        await log_callback("[error] stderr output:")
                        for sl in stderr_lines[-30:]:
                            await log_callback(f"  {sl}")
                return False

            return True

        except Exception as e:
            logger.exception(f"Subprocess failed: {cmd[0]}")
            if log_callback:
                await log_callback(f"[error] {e}")
            return False
        finally:
            task._process = None

    async def run_full_pipeline(
        self,
        task: TrainingTask,
        params: dict[str, Any],
        log_callback: Callable | None = None,
    ) -> bool:
        """Run the complete training pipeline."""
        exp_name = params["exp_name"]
        sr_label = params.get("sr", "40k")
        sr_int = {"32k": 32000, "40k": 40000, "48k": 48000}.get(sr_label, 40000)
        version = params.get("version", "v2")
        f0_method = params.get("f0_method", "rmvpe")
        input_dir = params.get("input_dir", "")

        # Step 1: Preprocess
        if log_callback:
            await log_callback("=" * 50)
            await log_callback("STEP 1/5: Preprocessing")
            await log_callback("=" * 50)

        if not await self.preprocess(task, exp_name, input_dir, sr_int, log_callback=log_callback):
            task.status = "error"
            return False
        if task._cancelled:
            return False

        # Step 2: F0 extraction
        if log_callback:
            await log_callback("=" * 50)
            await log_callback("STEP 2/5: F0 Extraction")
            await log_callback("=" * 50)

        if not await self.extract_f0(task, exp_name, f0_method, log_callback=log_callback):
            task.status = "error"
            return False
        if task._cancelled:
            return False

        # Step 3: Feature extraction
        if log_callback:
            await log_callback("=" * 50)
            await log_callback("STEP 3/5: Feature Extraction")
            await log_callback("=" * 50)

        if not await self.extract_features(task, exp_name, version, log_callback=log_callback):
            task.status = "error"
            return False
        if task._cancelled:
            return False

        # Step 3.5: Write filelist
        if not await self.write_filelist(exp_name, sr_int, version, log_callback=log_callback):
            task.status = "error"
            return False

        # Step 4: Training
        if log_callback:
            await log_callback("=" * 50)
            await log_callback("STEP 4/5: Model Training")
            await log_callback("=" * 50)

        if not await self.train(
            task,
            exp_name,
            sr=sr_label,
            f0=params.get("f0", True),
            batch_size=params.get("batch_size", 4),
            total_epoch=params.get("total_epoch", 100),
            save_epoch=params.get("save_epoch", 10),
            gpu_ids=params.get("gpu_ids", "0"),
            pretrained_g=params.get("pretrained_g", ""),
            pretrained_d=params.get("pretrained_d", ""),
            save_latest=params.get("save_latest", True),
            cache_gpu=params.get("cache_gpu", False),
            save_weights=params.get("save_weights", True),
            version=version,
            log_callback=log_callback,
        ):
            task.status = "error"
            return False
        if task._cancelled:
            return False

        # Step 5: Build index
        if log_callback:
            await log_callback("=" * 50)
            await log_callback("STEP 5/5: Building Index")
            await log_callback("=" * 50)

        if not await self.build_index(task, exp_name, version, log_callback=log_callback):
            task.status = "error"
            return False

        # Export model
        await self.export_model(exp_name, params.get("export_dir"), log_callback=log_callback)

        task.status = "completed"
        if log_callback:
            await log_callback("=" * 50)
            await log_callback("Training pipeline completed successfully!")
            await log_callback("=" * 50)

        return True


# Global instance
rvc_train_wrapper = RVCTrainWrapper()
