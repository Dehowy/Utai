"""UTAU voice bank pitch error + phoneme diagnostic.

Parses oto.ini, extracts audio segments, runs RMVPE for F0,
compares against labeled pitch. Also maps kana to IPA for
cross-dataset phoneme alignment comparison.

Usage: python scripts/diag_utau_pitch_phoneme.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'RVC/RVC20240604Nvidia')

import torch
import numpy as np
import os
import re
import time
import librosa
from collections import defaultdict
from infer.lib.rmvpe import RMVPE

UTAU_ROOT = 'kiritan/Utau'
RMVPE_PATH = 'RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt'

# ---- Note name to MIDI ----
NOTE_TO_MIDI = {}
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
for octave in range(0, 9):
    for i, name in enumerate(note_names):
        midi = (octave + 1) * 12 + i
        NOTE_TO_MIDI[f'{name}{octave}'] = midi
# Also handle flats if needed
NOTE_TO_MIDI['Db'] = NOTE_TO_MIDI.get('C#', 0)  # etc - but UTAU typically uses sharps

# ---- Kana to IPA mapping (romaji-based, simplified for singing) ----
KANA_TO_IPA = {
    # Vowels
    'あ': ['a'], 'い': ['i'], 'う': ['u'], 'え': ['e'], 'お': ['o'],
    # K-row
    'か': ['k', 'a'], 'き': ['k', 'i'], 'く': ['k', 'u'], 'け': ['k', 'e'], 'こ': ['k', 'o'],
    # S-row
    'さ': ['s', 'a'], 'し': ['ɕ', 'i'], 'す': ['s', 'u'], 'せ': ['s', 'e'], 'そ': ['s', 'o'],
    # T-row
    'た': ['t', 'a'], 'ち': ['tɕ', 'i'], 'つ': ['ts', 'u'], 'て': ['t', 'e'], 'と': ['t', 'o'],
    # N-row
    'な': ['n', 'a'], 'に': ['ɲ', 'i'], 'ぬ': ['n', 'u'], 'ね': ['n', 'e'], 'の': ['n', 'o'],
    # H-row
    'は': ['h', 'a'], 'ひ': ['ç', 'i'], 'ふ': ['ɸ', 'u'], 'へ': ['h', 'e'], 'ほ': ['h', 'o'],
    # M-row
    'ま': ['m', 'a'], 'み': ['m', 'i'], 'む': ['m', 'u'], 'め': ['m', 'e'], 'も': ['m', 'o'],
    # Y-row
    'や': ['j', 'a'], 'ゆ': ['j', 'u'], 'よ': ['j', 'o'],
    # R-row
    'ら': ['ɾ', 'a'], 'り': ['ɾ', 'i'], 'る': ['ɾ', 'u'], 'れ': ['ɾ', 'e'], 'ろ': ['ɾ', 'o'],
    # W-row
    'わ': ['w', 'a'], 'を': ['o'],
    # N
    'ん': ['n'],
    # G-row (voiced)
    'が': ['g', 'a'], 'ぎ': ['g', 'i'], 'ぐ': ['g', 'u'], 'げ': ['g', 'e'], 'ご': ['g', 'o'],
    # Z-row
    'ざ': ['z', 'a'], 'じ': ['dʑ', 'i'], 'ず': ['z', 'u'], 'ぜ': ['z', 'e'], 'ぞ': ['z', 'o'],
    # D-row
    'だ': ['d', 'a'], 'ぢ': ['dʑ', 'i'], 'づ': ['z', 'u'], 'で': ['d', 'e'], 'ど': ['d', 'o'],
    # B-row
    'ば': ['b', 'a'], 'び': ['b', 'i'], 'ぶ': ['b', 'u'], 'べ': ['b', 'e'], 'ぼ': ['b', 'o'],
    # P-row
    'ぱ': ['p', 'a'], 'ぴ': ['p', 'i'], 'ぷ': ['p', 'u'], 'ぺ': ['p', 'e'], 'ぽ': ['p', 'o'],
    # Small kana (拗音 combinations)
    'ゃ': ['j', 'a'], 'ゅ': ['j', 'u'], 'ょ': ['j', 'o'],
    'ぃ': ['i'], 'ぅ': ['u'], 'ぇ': ['e'], 'ぉ': ['o'], 'ぁ': ['a'],
    # Special
    'うぃ': ['w', 'i'], 'うぇ': ['w', 'e'], 'うぉ': ['w', 'o'],
    'いぇ': ['j', 'e'],
}

# UTAU romaji phonemes (from oto.ini alias: "a い" means prev=a, curr=い)
ROMAJI_TO_IPA = {
    'a': 'a', 'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o',
    'k': 'k', 's': 's', 't': 't', 'n': 'n', 'h': 'h',
    'm': 'm', 'y': 'j', 'r': 'ɾ', 'w': 'w',
    'g': 'g', 'z': 'z', 'd': 'd', 'b': 'b', 'p': 'p',
    'N': 'n', 'J': 'dʑ',
}


def parse_pitch_from_alias(alias):
    """Extract MIDI note number from oto.ini alias suffix like 'D4', 'C5', 'G4'."""
    # Match note name at end: C4, D#4, G4, etc.
    m = re.search(r'([A-G]#?)(\d)$', alias.strip())
    if m:
        note = m.group(1) + m.group(2)
        return NOTE_TO_MIDI.get(note, 0)
    return 0


def parse_phoneme_from_alias(alias):
    """Extract the target phoneme from alias.
    Formats: '- あD4' (initial), 'a いD4' (transition), 'a RD4' (release)
    Returns the target kana/romaji and its IPA vowel.
    """
    # Remove pitch suffix
    cleaned = re.sub(r'[A-G]#?\d$', '', alias.strip())
    # Remove number suffixes like あ2
    cleaned = re.sub(r'\d+$', '', cleaned)

    parts = cleaned.split()
    if len(parts) >= 2:
        target = parts[-1]  # the target phoneme/kana
    elif len(parts) == 1:
        target = parts[0]
    else:
        return None, None

    # Skip special: R (release), ・ (glottal), 息 (breath)
    if target in ('R', '・', '-') or target.startswith('息'):
        return None, None

    # Try kana lookup
    if target in KANA_TO_IPA:
        ipa_seq = KANA_TO_IPA[target]
        vowel = ipa_seq[-1] if ipa_seq else None  # last element is typically the vowel
        return target, vowel

    # Single romaji vowel
    if target in ROMAJI_TO_IPA:
        return target, ROMAJI_TO_IPA[target]

    return target, None


def parse_oto_ini(oto_path):
    """Parse oto.ini file. Returns list of (wav_path, alias, offset_ms, consonant_ms, cutoff_ms, preutterance_ms, overlap_ms)."""
    entries = []
    oto_dir = os.path.dirname(oto_path)
    for enc in ['shift_jis', 'cp932', 'utf-8', 'utf-8-sig']:
        try:
            with open(oto_path, encoding=enc) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return entries

    for line in lines:
        line = line.strip()
        if not line or '=' not in line:
            continue
        try:
            wav_part, params = line.split('=', 1)
            parts = params.split(',')
            if len(parts) < 6:
                continue
            alias = parts[0].strip()
            offset = float(parts[1])
            consonant = float(parts[2])
            cutoff = float(parts[3])
            preutterance = float(parts[4])
            overlap = float(parts[5])
            wav_path = os.path.join(oto_dir, wav_part.strip())
            entries.append({
                'wav': wav_path,
                'alias': alias,
                'offset_ms': offset,
                'consonant_ms': consonant,
                'cutoff_ms': cutoff,
                'preutterance_ms': preutterance,
                'overlap_ms': overlap,
            })
        except (ValueError, IndexError):
            continue
    return entries


def load_segment(wav_path, offset_ms, cutoff_ms, sr=16000):
    """Load a segment of a WAV file based on oto.ini timing."""
    try:
        wav, orig_sr = librosa.load(wav_path, sr=sr, mono=True)
    except Exception:
        return None

    total_ms = len(wav) / sr * 1000
    start_ms = max(0, offset_ms)
    if cutoff_ms < 0:
        end_ms = total_ms + cutoff_ms  # cutoff is negative = from end
    else:
        end_ms = start_ms + cutoff_ms if cutoff_ms > 0 else total_ms

    start_sample = int(start_ms / 1000 * sr)
    end_sample = int(end_ms / 1000 * sr)
    end_sample = min(end_sample, len(wav))

    if end_sample <= start_sample or end_sample - start_sample < sr * 0.05:  # min 50ms
        return None

    return wav[start_sample:end_sample]


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# ---- Main ----
print('Loading RMVPE...', flush=True)
rmvpe = RMVPE(RMVPE_PATH, is_half=False, device='cuda')
print('RMVPE loaded.\n', flush=True)

# Collect all oto.ini entries across all banks
all_banks = {}
for bank_name in sorted(os.listdir(UTAU_ROOT)):
    bank_path = os.path.join(UTAU_ROOT, bank_name)
    if not os.path.isdir(bank_path):
        continue

    entries = []
    for root, dirs, files in os.walk(bank_path):
        for f in files:
            if f.lower() == 'oto.ini':
                entries.extend(parse_oto_ini(os.path.join(root, f)))
    all_banks[bank_name] = entries
    print(f'  {bank_name}: {len(entries)} oto entries', flush=True)

print(f'\nTotal: {sum(len(e) for e in all_banks.values())} entries across {len(all_banks)} banks\n', flush=True)

# ---- Process each bank ----
overall_cents = []
overall_abs_cents = []
per_bank_stats = {}
per_vowel_cents = defaultdict(list)  # IPA vowel -> list of cent errors
n_processed = 0
n_skipped = 0
n_no_pitch = 0
n_no_voiced = 0

t0 = time.time()

for bank_name, entries in all_banks.items():
    bank_cents = []
    bank_abs_cents = []

    for entry in entries:
        pitch_midi = parse_pitch_from_alias(entry['alias'])
        if pitch_midi == 0:
            n_no_pitch += 1
            continue

        kana, ipa_vowel = parse_phoneme_from_alias(entry['alias'])
        if kana is None:
            n_skipped += 1
            continue

        # Load audio segment
        segment = load_segment(entry['wav'], entry['offset_ms'], entry['cutoff_ms'])
        if segment is None or len(segment) < 800:  # min 50ms at 16kHz
            n_skipped += 1
            continue

        # RMVPE F0
        f0 = rmvpe.infer_from_audio(
            torch.from_numpy(segment).float().to('cuda'), thred=0.03
        )
        voiced = f0[f0 > 30.0]
        if len(voiced) < 3:
            n_no_voiced += 1
            continue

        # Cent error
        expected_hz = midi_to_hz(pitch_midi)
        cents = 1200.0 * np.log2(voiced / expected_hz)
        abs_cents = np.abs(cents)

        bank_cents.extend(cents.tolist())
        bank_abs_cents.extend(abs_cents.tolist())
        overall_cents.extend(cents.tolist())
        overall_abs_cents.extend(abs_cents.tolist())

        if ipa_vowel:
            per_vowel_cents[ipa_vowel].extend(cents.tolist())

        n_processed += 1

    if bank_cents:
        bc = np.array(bank_cents)
        ba = np.array(bank_abs_cents)
        per_bank_stats[bank_name] = {
            'n': len(bank_cents),
            'signed_mean': bc.mean(),
            'signed_median': np.median(bc),
            'abs_mean': ba.mean(),
            'abs_median': np.median(ba),
            'abs_p90': np.percentile(ba, 90),
            'pct_100': 100 * (ba > 100).sum() / len(ba),
            'pct_200': 100 * (ba > 200).sum() / len(ba),
        }

    elapsed = time.time() - t0
    print(f'  {bank_name}: {n_processed} processed, {elapsed:.0f}s', flush=True)

# ---- Results ----
print('\n' + '=' * 100)
print('UTAU PITCH ERROR: RMVPE F0 vs labeled pitch (cents)')
print('=' * 100)

print(f'\nProcessed: {n_processed} entries, Skipped: {n_skipped}, No pitch: {n_no_pitch}, No voiced: {n_no_voiced}\n')

print(f'{"Bank":30s} {"Frames":>8s} {"SignMean":>9s} {"SignMed":>8s} {"AbsMean":>8s} {"AbsMed":>7s} {"P90":>7s} {">100c":>7s} {">200c":>7s}')
print('-' * 100)
for bank_name in sorted(per_bank_stats.keys()):
    s = per_bank_stats[bank_name]
    print(f'{bank_name:30s} {s["n"]:>8d} {s["signed_mean"]:>8.1f}c {s["signed_median"]:>7.1f}c '
          f'{s["abs_mean"]:>7.1f}c {s["abs_median"]:>6.1f}c {s["abs_p90"]:>6.1f}c '
          f'{s["pct_100"]:>6.1f}% {s["pct_200"]:>6.1f}%')

# Overall
if overall_abs_cents:
    oc = np.array(overall_cents)
    oa = np.array(overall_abs_cents)
    print(f'\n{"OVERALL":30s} {len(oa):>8d} {oc.mean():>8.1f}c {np.median(oc):>7.1f}c '
          f'{oa.mean():>7.1f}c {np.median(oa):>6.1f}c {np.percentile(oa,90):>6.1f}c '
          f'{100*(oa>100).sum()/len(oa):>6.1f}% {100*(oa>200).sum()/len(oa):>6.1f}%')

    # Distribution
    print(f'\nDistribution (overall):')
    bins = [0, 10, 25, 50, 100, 200, 500, 1000, 99999]
    labels = ['0-10c', '10-25c', '25-50c', '50-100c', '100-200c', '200-500c', '500-1000c', '>1000c']
    hist, _ = np.histogram(oa, bins=bins)
    for label, count in zip(labels, hist):
        pct = 100.0 * count / len(oa)
        bar = '#' * int(pct / 2)
        print(f'  {label:>10s}: {count:>8d} ({pct:>5.1f}%) {bar}')

# Per-vowel
print(f'\n{"=" * 80}')
print('Per-vowel pitch error (IPA)')
print(f'{"=" * 80}')
for vowel in ['a', 'i', 'u', 'e', 'o']:
    if vowel in per_vowel_cents:
        vc = np.array(per_vowel_cents[vowel])
        va = np.abs(vc)
        print(f'  /{vowel}/: n={len(vc):>6d}  signed_mean={vc.mean():>6.1f}c  '
              f'abs_mean={va.mean():>5.1f}c  abs_median={np.median(va):>5.1f}c  '
              f'>100c={100*(va>100).sum()/len(va):>5.1f}%')

# Comparison with preprocessed datasets
print(f'\n{"=" * 80}')
print('COMPARISON with preprocessed datasets')
print(f'{"=" * 80}')
print(f'{"Dataset":20s} {"AbsMean":>8s} {"AbsMedian":>10s} {"P90":>8s} {">100c":>8s} {">200c":>8s}')
print('-' * 60)
print(f'{"ACE-Opencpop":20s} {"140.0c":>8s} {"23.9c":>10s} {"378.4c":>8s} {"31.1%":>8s} {"20.4%":>8s}')
print(f'{"GTSinger JA":20s} {"90.0c":>8s} {"29.5c":>10s} {"199.5c":>8s} {"18.0%":>8s} {"10.0%":>8s}')
print(f'{"GTSinger EN":20s} {"85.7c":>8s} {"32.0c":>10s} {"179.6c":>8s} {"16.8%":>8s} {"8.8%":>8s}')
if overall_abs_cents:
    oa = np.array(overall_abs_cents)
    print(f'{"UTAU (this run)":20s} {oa.mean():>7.1f}c {np.median(oa):>9.1f}c '
          f'{np.percentile(oa,90):>7.1f}c {100*(oa>100).sum()/len(oa):>7.1f}% '
          f'{100*(oa>200).sum()/len(oa):>7.1f}%')

print('\nDone.', flush=True)
