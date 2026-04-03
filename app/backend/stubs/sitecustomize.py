"""Auto-loaded by Python at startup (via PYTHONPATH).
Patches torch.utils.tensorboard with a no-op SummaryWriter so RVC training
scripts work without the real tensorboard package installed."""

import importlib
import sys
import types


def _install_tensorboard_stub():
    """Install a fake tensorboard + torch.utils.tensorboard module."""
    # Create stub tensorboard top-level
    tb = types.ModuleType("tensorboard")
    tb.__version__ = "2.0.0"
    sys.modules["tensorboard"] = tb

    # Stub submodules that torch.utils.tensorboard.writer imports
    for sub in [
        "tensorboard.compat",
        "tensorboard.compat.proto",
        "tensorboard.compat.proto.summary_pb2",
        "tensorboard.compat.tf",
        "tensorboard.plugins",
        "tensorboard.plugins.projector",
        "tensorboard.plugins.projector.plugin_data_pb2",
        "tensorboard.summary",
        "tensorboard.summary.writer",
        "tensorboard.summary.writer.record_writer",
    ]:
        mod = types.ModuleType(sub)
        # Make tf stub behave like a module with common attrs
        if sub == "tensorboard.compat.tf":
            mod.io = types.ModuleType("tensorboard.compat.tf.io")
            mod.io.gfile = types.ModuleType("tensorboard.compat.tf.io.gfile")
            mod.io.gfile.exists = lambda path: False
            mod.io.gfile.makedirs = lambda path: None
            sys.modules["tensorboard.compat.tf.io"] = mod.io
            sys.modules["tensorboard.compat.tf.io.gfile"] = mod.io.gfile
        sys.modules[sub] = mod
        # Set as attr on parent
        parts = sub.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], parts[-1], mod)

    # Stub RecordWriter
    class RecordWriter:
        def __init__(self, *a, **kw): pass
        def write(self, *a, **kw): pass
        def flush(self): pass
        def close(self): pass

    sys.modules["tensorboard.summary.writer.record_writer"].RecordWriter = RecordWriter

    # Now create a no-op SummaryWriter and patch torch.utils.tensorboard
    class SummaryWriter:
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def add_scalars(self, *a, **kw): pass
        def add_image(self, *a, **kw): pass
        def add_images(self, *a, **kw): pass
        def add_audio(self, *a, **kw): pass
        def add_text(self, *a, **kw): pass
        def add_histogram(self, *a, **kw): pass
        def add_figure(self, *a, **kw): pass
        def flush(self): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    FileWriter = SummaryWriter  # alias

    # Pre-populate torch.utils.tensorboard in sys.modules
    tb_torch = types.ModuleType("torch.utils.tensorboard")
    tb_torch.SummaryWriter = SummaryWriter
    tb_torch.FileWriter = FileWriter
    sys.modules["torch.utils.tensorboard"] = tb_torch


_install_tensorboard_stub()
