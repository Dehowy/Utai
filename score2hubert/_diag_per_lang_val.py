"""Quick per-language val metrics for the current best.pt checkpoint."""
import sys, json, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from eval_audio import load_model

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/finetune_discrete_balanced/best.pt"
proc = Path("data/processed/multi_lang_4")
device = "cuda"

print(f"Loading {CKPT}")
model, _ = load_model(CKPT, device)

md = json.loads((proc / "metadata.json").read_text(encoding="utf-8"))
val = [m for m in md if m.get("split") == "val"]

lang_metrics = defaultdict(lambda: {"l1": [], "cos": [], "n": 0})

for i, m in enumerate(val):
    sid, lang = m["id"], m.get("language", "zh")
    hubert_gt = torch.load(proc / "hubert" / f"{sid}.pt", weights_only=True).float()
    phonemes = torch.load(proc / "phonemes" / f"{sid}.pt", weights_only=True).long()
    pitch = torch.load(proc / "pitch" / f"{sid}.pt", weights_only=True).long()
    duration = torch.load(proc / "duration" / f"{sid}.pt", weights_only=True).long()
    tech_p = proc / "tech" / f"{sid}.pt"
    tech = torch.load(tech_p, weights_only=True).float() if tech_p.exists() else torch.zeros(phonemes.size(0), 6)
    
    with torch.no_grad():
        out = model.synthesize(
            phonemes=phonemes.unsqueeze(0).to(device),
            pitch=pitch.unsqueeze(0).to(device),
            duration=duration.unsqueeze(0).to(device),
            singing_ratio=torch.tensor([m.get("singing_ratio", 1.0)], device=device),
            lang_id=torch.tensor([m.get("lang_id", 0)], dtype=torch.long, device=device),
            tech=tech.unsqueeze(0).to(device),
            n_steps=1, solver="euler", cfg_scale=1.0, temperature=1.0,
        )
    h = out["hubert"][0].cpu()
    T = min(h.size(0), hubert_gt.size(0))
    l1 = F.l1_loss(h[:T], hubert_gt[:T]).item()
    cos = F.cosine_similarity(h[:T], hubert_gt[:T], dim=-1).mean().item()
    lang_metrics[lang]["l1"].append(l1)
    lang_metrics[lang]["cos"].append(cos)
    lang_metrics[lang]["n"] += 1
    
    if (i + 1) % 500 == 0:
        print(f"  processed {i+1}/{len(val)}")

print(f"\n{'lang':>4s} {'n':>6s} {'l1':>8s} {'cos':>8s}")
print("-" * 32)
for lang in sorted(lang_metrics.keys()):
    d = lang_metrics[lang]
    ml1 = sum(d["l1"]) / len(d["l1"])
    mcos = sum(d["cos"]) / len(d["cos"])
    print(f"{lang:>4s} {d['n']:6d} {ml1:8.4f} {mcos:8.4f}")

all_l1 = [v for d in lang_metrics.values() for v in d["l1"]]
all_cos = [v for d in lang_metrics.values() for v in d["cos"]]
print(f" all {len(all_l1):6d} {sum(all_l1)/len(all_l1):8.4f} {sum(all_cos)/len(all_cos):8.4f}")
