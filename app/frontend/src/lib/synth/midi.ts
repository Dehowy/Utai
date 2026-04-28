/** MIDI (.mid) binary parser — converts Standard MIDI Files to UProject format. */

import { RESOLUTION } from '../../types/ustx';
import type {
  UProject, UNote, UInstrumentNote, UVoicePart, UInstrumentPart,
  UTrack, UTimeSignature, UExpressionDescriptor,
} from '../../types/ustx';
import { defaultPitch, defaultVibrato } from '../../types/ustx';

/* ── Internal types ── */

interface MidiNote {
  tick: number;
  channel: number;
  note: number;
  velocity: number;
  duration: number;
}

interface MidiLyric {
  tick: number;
  text: string;
}

interface MidiTrack {
  name: string;
  channel: number;
  notes: MidiNote[];
  lyrics: MidiLyric[];
  tempos: { tick: number; bpm: number }[];
  timeSigs: { tick: number; num: number; den: number }[];
  programs: Map<number, number>;
}

/* ── Binary helpers ── */

function readVLQ(view: DataView, offset: number): [number, number] {
  let value = 0;
  let bytes = 0;
  let b: number;
  do {
    b = view.getUint8(offset + bytes);
    value = (value << 7) | (b & 0x7F);
    bytes++;
  } while (b & 0x80);
  return [value, bytes];
}

function decodeText(bytes: Uint8Array): string {
  try { return new TextDecoder('utf-8', { fatal: true }).decode(bytes); } catch { /* noop */ }
  try { return new TextDecoder('gbk', { fatal: true }).decode(bytes); } catch { /* noop */ }
  try { return new TextDecoder('shift-jis', { fatal: true }).decode(bytes); } catch { /* noop */ }
  return new TextDecoder('utf-8').decode(bytes);
}

/* ── MIDI binary parser ── */

function parseMidiFile(buf: ArrayBuffer): {
  format: number; tpb: number; tracks: MidiTrack[];
} {
  const view = new DataView(buf);
  const raw = new Uint8Array(buf);

  if (String.fromCharCode(raw[0], raw[1], raw[2], raw[3]) !== 'MThd')
    throw new Error('Not a valid MIDI file');

  const headerLen = view.getUint32(4);
  const format = view.getUint16(8);
  const numTracks = view.getUint16(10);
  const tpb = view.getUint16(12);
  if (tpb & 0x8000) throw new Error('SMPTE time division not supported');

  const tracks: MidiTrack[] = [];
  let pos = 8 + headerLen;

  for (let t = 0; t < numTracks; t++) {
    if (pos + 8 > raw.length) break;
    if (String.fromCharCode(raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]) !== 'MTrk')
      throw new Error(`Expected MTrk at offset ${pos}`);
    const chunkLen = view.getUint32(pos + 4);
    const end = pos + 8 + chunkLen;
    pos += 8;

    const track: MidiTrack = {
      name: '', channel: -1, notes: [], lyrics: [],
      tempos: [], timeSigs: [], programs: new Map(),
    };

    let tick = 0;
    let running = 0;
    const active = new Map<string, { tick: number; velocity: number }>();
    const chCounts = new Map<number, number>();
    const bump = (ch: number) => chCounts.set(ch, (chCounts.get(ch) || 0) + 1);

    while (pos < end) {
      const [delta, db] = readVLQ(view, pos);
      pos += db;
      tick += delta;

      let status = raw[pos];
      if (status < 0x80) {
        status = running;
      } else {
        pos++;
        if (status < 0xF0) running = status;
      }

      const type = status & 0xF0;
      const ch = status & 0x0F;

      if (type === 0x80) {
        const note = raw[pos++];
        pos++; // velocity (unused for note-off)
        const key = `${ch}-${note}`;
        const start = active.get(key);
        if (start) {
          track.notes.push({ tick: start.tick, channel: ch, note, velocity: start.velocity, duration: tick - start.tick });
          active.delete(key);
        }
        bump(ch);
      } else if (type === 0x90) {
        const note = raw[pos++];
        const vel = raw[pos++];
        const key = `${ch}-${note}`;
        if (vel === 0) {
          const start = active.get(key);
          if (start) {
            track.notes.push({ tick: start.tick, channel: ch, note, velocity: start.velocity, duration: tick - start.tick });
            active.delete(key);
          }
        } else {
          const existing = active.get(key);
          if (existing) {
            track.notes.push({ tick: existing.tick, channel: ch, note, velocity: existing.velocity, duration: tick - existing.tick });
          }
          active.set(key, { tick, velocity: vel });
        }
        bump(ch);
      } else if (type === 0xA0) { pos += 2; }
      else if (type === 0xB0) { pos += 2; }
      else if (type === 0xC0) { track.programs.set(ch, raw[pos++]); bump(ch); }
      else if (type === 0xD0) { pos += 1; }
      else if (type === 0xE0) { pos += 2; }
      else if (status === 0xFF) {
        const mt = raw[pos++];
        const [ml, mlb] = readVLQ(view, pos);
        pos += mlb;
        if (mt === 0x03) {
          track.name = decodeText(raw.slice(pos, pos + ml));
        } else if (mt === 0x01 || mt === 0x05) {
          const txt = decodeText(raw.slice(pos, pos + ml)).replace(/[\r\n]/g, '').trim();
          if (txt) track.lyrics.push({ tick, text: txt });
        } else if (mt === 0x51 && ml === 3) {
          const uspb = (raw[pos] << 16) | (raw[pos + 1] << 8) | raw[pos + 2];
          track.tempos.push({ tick, bpm: Math.round(6e7 / uspb * 100) / 100 });
        } else if (mt === 0x58 && ml >= 2) {
          track.timeSigs.push({ tick, num: raw[pos], den: 1 << raw[pos + 1] });
        }
        pos += ml;
      } else if (status === 0xF0 || status === 0xF7) {
        const [sl, slb] = readVLQ(view, pos);
        pos += slb + sl;
      } else if (status >= 0xF1 && status <= 0xF6) {
        if (status === 0xF2) pos += 2;
        else if (status === 0xF1 || status === 0xF3) pos += 1;
      }
    }

    // close any still-active notes at track end
    for (const [key, start] of active) {
      const [cs, ns] = key.split('-');
      track.notes.push({ tick: start.tick, channel: +cs, note: +ns, velocity: start.velocity, duration: tick - start.tick });
    }

    // determine primary channel
    let maxC = 0;
    for (const [c, cnt] of chCounts) {
      if (cnt > maxC) { maxC = cnt; track.channel = c; }
    }

    pos = end;
    tracks.push(track);
  }

  return { format, tpb, tracks };
}

/* ── Conversion helpers ── */

const TRACK_COLORS = ['Blue', 'Pink', 'Green', 'Yellow', 'Red', 'Cyan', 'Purple', 'Orange'];

const SPECIAL_LYRICS = new Set([
  'R', 'r', '-', '+', '+~', '++',
  'br', 'BR', 'breath', 'Breath',
  'AP', 'ap', 'SP', 'sp', 'pau', 'sil',
]);

const DEFAULT_EXPRESSIONS: Record<string, UExpressionDescriptor> = {
  dyn: { name: 'dynamics (curve)', abbr: 'dyn', type: 2, min: -240, max: 120, default_value: 0 },
  pitd: { name: 'pitch deviation (curve)', abbr: 'pitd', type: 2, min: -1200, max: 1200, default_value: 0 },
  clr: { name: 'voice color', abbr: 'clr', type: 1, min: 0, max: 0, default_value: 0, options: [''] },
  eng: { name: 'resampler engine', abbr: 'eng', type: 1, min: 0, max: 0, default_value: 0, options: [''] },
  vel: { name: 'velocity', abbr: 'vel', type: 0, min: 0, max: 200, default_value: 100 },
  vol: { name: 'volume', abbr: 'vol', type: 0, min: 0, max: 200, default_value: 100 },
  atk: { name: 'attack', abbr: 'atk', type: 0, min: 0, max: 200, default_value: 100 },
  dec: { name: 'decay', abbr: 'dec', type: 0, min: 0, max: 100, default_value: 0 },
  gen: { name: 'gender', abbr: 'gen', type: 0, min: -100, max: 100, default_value: 0 },
  bre: { name: 'breath', abbr: 'bre', type: 0, min: 0, max: 100, default_value: 0 },
};

function timeSigsToBarPositions(
  timeSigs: { tick: number; num: number; den: number }[],
): UTimeSignature[] {
  if (!timeSigs.length) return [{ bar_position: 0, beat_per_bar: 4, beat_unit: 4 }];
  const sorted = [...timeSigs].sort((a, b) => a.tick - b.tick);
  if (sorted[0].tick > 0) sorted.unshift({ tick: 0, num: 4, den: 4 });

  const result: UTimeSignature[] = [];
  let barStart = 0;
  let bar = 0;

  for (let i = 0; i < sorted.length; i++) {
    if (i > 0) {
      const prev = sorted[i - 1];
      const ticksPerBar = prev.num * (RESOLUTION * 4 / prev.den);
      const elapsed = Math.floor((sorted[i].tick - barStart) / ticksPerBar);
      bar += elapsed;
      barStart += elapsed * ticksPerBar;
    }
    result.push({ bar_position: bar, beat_per_bar: sorted[i].num, beat_unit: sorted[i].den });
  }
  return result;
}

function pairLyricsToNotes(
  notes: MidiNote[],
  lyrics: MidiLyric[],
  tpb: number,
): Map<MidiNote, string> {
  const map = new Map<MidiNote, string>();
  if (!lyrics.length || !notes.length) return map;

  // Sequential pairing — works for the common 1:1 lyric-to-note case
  if (lyrics.length <= notes.length * 1.5) {
    for (let i = 0; i < Math.min(lyrics.length, notes.length); i++) {
      map.set(notes[i], lyrics[i].text);
    }
    return map;
  }

  // Timing-based fallback
  let ni = 0;
  for (const l of lyrics) {
    while (ni < notes.length && notes[ni].tick + notes[ni].duration < l.tick) ni++;
    if (ni < notes.length && Math.abs(notes[ni].tick - l.tick) <= tpb) {
      map.set(notes[ni], l.text);
      ni++;
    }
  }
  return map;
}

function detectLanguage(lyrics: MidiLyric[]): 'zh' | 'ja' {
  const all = lyrics.map(l => l.text).join('');
  let ja = 0;
  for (const ch of all) {
    const cp = ch.codePointAt(0) || 0;
    if ((cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF)) ja++;
  }
  return ja / Math.max(all.length, 1) > 0.3 ? 'ja' : 'zh';
}

/* ── Main export ── */

interface TrackGroup {
  name: string;
  channel: number;
  notes: MidiNote[];
  lyrics: MidiLyric[];
}

export function parseMidi(buf: ArrayBuffer, filename?: string): UProject {
  const { format, tpb, tracks } = parseMidiFile(buf);
  const scale = RESOLUTION / tpb;

  // collect tempo / time-sig from all tracks (scaled to RESOLUTION)
  const allTempos: { tick: number; bpm: number }[] = [];
  const allTimeSigs: { tick: number; num: number; den: number }[] = [];
  for (const tr of tracks) {
    for (const t of tr.tempos) allTempos.push({ tick: Math.round(t.tick * scale), bpm: t.bpm });
    for (const ts of tr.timeSigs) allTimeSigs.push({ tick: Math.round(ts.tick * scale), num: ts.num, den: ts.den });
  }
  allTempos.sort((a, b) => a.tick - b.tick);
  allTimeSigs.sort((a, b) => a.tick - b.tick);

  // group notes by track/channel
  const groups: TrackGroup[] = [];

  if (format === 0 && tracks.length >= 1) {
    // Type 0: single track — split by channel
    const tr = tracks[0];
    const byChannel = new Map<number, MidiNote[]>();
    for (const n of tr.notes) {
      if (!byChannel.has(n.channel)) byChannel.set(n.channel, []);
      byChannel.get(n.channel)!.push(n);
    }

    // vocal channel: the one whose notes best align with lyrics
    let vocalCh = -1;
    if (tr.lyrics.length > 0) {
      let best = 0;
      for (const [ch, chNotes] of byChannel) {
        const sorted = [...chNotes].sort((a, b) => a.tick - b.tick);
        let matches = 0, ni = 0;
        for (const l of tr.lyrics) {
          while (ni < sorted.length && sorted[ni].tick + sorted[ni].duration < l.tick) ni++;
          if (ni < sorted.length && Math.abs(sorted[ni].tick - l.tick) <= tpb) { matches++; ni++; }
        }
        if (matches > best) { best = matches; vocalCh = ch; }
      }
    }

    for (const [ch, chNotes] of byChannel) {
      groups.push({
        name: ch === 9 ? 'Drums' : ch === vocalCh ? 'Vocal' : `Channel ${ch + 1}`,
        channel: ch,
        notes: chNotes,
        lyrics: ch === vocalCh ? tr.lyrics : [],
      });
    }
  } else {
    // Type 1/2: per-track, with orphan-lyrics merging
    const orphans: { name: string; lyrics: MidiLyric[] }[] = [];
    for (const tr of tracks) {
      if (tr.notes.length > 0) {
        groups.push({ name: tr.name || `Track ${groups.length + 1}`, channel: tr.channel, notes: tr.notes, lyrics: tr.lyrics });
      } else if (tr.lyrics.length > 0) {
        orphans.push({ name: tr.name, lyrics: tr.lyrics });
      }
    }

    // merge orphan lyrics into best-matching notes group
    for (const o of orphans) {
      let bestIdx = -1, bestScore = 0;
      for (let i = 0; i < groups.length; i++) {
        if (groups[i].lyrics.length > 0) continue;
        const sorted = [...groups[i].notes].sort((a, b) => a.tick - b.tick);
        let score = 0, ni = 0;
        for (const l of o.lyrics) {
          while (ni < sorted.length && sorted[ni].tick + sorted[ni].duration < l.tick) ni++;
          if (ni < sorted.length && Math.abs(sorted[ni].tick - l.tick) <= tpb) { score++; ni++; }
        }
        if (score > bestScore) { bestScore = score; bestIdx = i; }
      }
      if (bestIdx >= 0) {
        groups[bestIdx].lyrics = o.lyrics;
        if (o.name) groups[bestIdx].name = o.name;
      }
    }
  }

  // sort: vocal tracks first
  groups.sort((a, b) => (a.lyrics.length > 0 ? 0 : 1) - (b.lyrics.length > 0 ? 0 : 1));

  // build UProject
  const uTracks: UTrack[] = [];
  const voiceParts: UVoicePart[] = [];
  const instrumentParts: UInstrumentPart[] = [];

  for (const g of groups) {
    const sorted = [...g.notes].sort((a, b) => a.tick - b.tick || a.note - b.note);
    if (!sorted.length) continue;

    const isVocal = g.lyrics.length > 0;
    const trackIdx = uTracks.length;

    uTracks.push({
      track_name: g.name,
      track_color: TRACK_COLORS[trackIdx % TRACK_COLORS.length],
      track_type: isVocal ? 'vocal' : 'instrument',
      mute: false, solo: false, volume: 0, pan: 0,
      voice_color_names: [''],
    });

    const minTick = sorted[0].tick;
    const maxEnd = sorted.reduce((m, n) => Math.max(m, n.tick + n.duration), 0);
    const partPos = Math.round(minTick * scale);
    const partDur = Math.max(Math.round(maxEnd * scale) - partPos, RESOLUTION);

    if (isVocal) {
      const lyricMap = pairLyricsToNotes(sorted, g.lyrics, tpb);
      const lang = detectLanguage(g.lyrics);
      const defaultLyric = lang === 'ja' ? 'あ' : '啊';

      const notes: UNote[] = [];
      for (const n of sorted) {
        const lyric = lyricMap.get(n)?.trim() ?? '';
        if (SPECIAL_LYRICS.has(lyric)) continue;
        notes.push({
          position: Math.round(n.tick * scale) - partPos,
          duration: Math.max(Math.round(n.duration * scale), 10),
          tone: n.note,
          lyric: lyric || defaultLyric,
          pitch: defaultPitch(),
          vibrato: defaultVibrato(),
          tuning: 0,
          phoneme_expressions: [],
          phoneme_overrides: [],
        });
      }
      voiceParts.push({
        name: g.name, comment: '', track_no: trackIdx,
        position: partPos, duration: partDur, notes, curves: [],
      });
    } else {
      const notes: UInstrumentNote[] = sorted.map(n => ({
        position: Math.round(n.tick * scale) - partPos,
        duration: Math.max(Math.round(n.duration * scale), 10),
        tone: n.note,
        velocity: n.velocity,
      }));
      instrumentParts.push({
        name: g.name, track_no: trackIdx,
        position: partPos, duration: partDur, notes,
      });
    }
  }

  // fallback: ensure at least one track
  if (!uTracks.length) {
    uTracks.push({
      track_name: 'Track1', track_color: 'Blue', track_type: 'vocal',
      mute: false, solo: false, volume: 0, pan: 0, voice_color_names: [''],
    });
  }

  return {
    name: filename || 'MIDI Import',
    comment: '',
    output_dir: 'Vocal',
    cache_dir: 'UCache',
    ustx_version: '0.9',
    key: 0,
    expressions: { ...DEFAULT_EXPRESSIONS },
    exp_selectors: Object.keys(DEFAULT_EXPRESSIONS),
    exp_primary: 0,
    exp_secondary: 1,
    time_signatures: timeSigsToBarPositions(allTimeSigs),
    tempos: allTempos.length
      ? allTempos.map(t => ({ position: t.tick, bpm: t.bpm }))
      : [{ position: 0, bpm: 120 }],
    tracks: uTracks,
    voice_parts: voiceParts,
    wave_parts: [],
    instrument_parts: instrumentParts,
  };
}
