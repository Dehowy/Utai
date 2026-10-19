import os

import torch
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary

# PyTorch >= 2.6 changed the default of `weights_only` in torch.load() from
# False to True. fairseq 0.12.2 (used to load the HuBERT checkpoint) does not
# pass `weights_only`, so loading assets/hubert/hubert_base.pt fails with
# "Unsupported global: GLOBAL fairseq.data.dictionary.Dictionary".
# Allowlist the one fairseq helper class used by the checkpoint so HuBERT can
# still be loaded without turning weights_only protection off. No-op on
# torch < 2.6.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([Dictionary])


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
