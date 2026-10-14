"""Voice-first lip-sync pipeline: the voice take is the master timeline.

    script line ──► (optional) ElevenLabs TTS ──► take.wav
                                                    │
                    Whisper (offline) ◄─────────────┤──► Rhubarb (+ script as dialog hint)
                          │                         │            │
                    QA gate: transcript vs script   │      mouth cues A–H, X
                          └──────────────► timeline.json ◄───────┘

Outputs mouth-shape *data*, not video frames, so any rig (LATCH, Godot/Unity Rhubarb
plugins, After Effects, our animate.py renderer) can play it in sync with the audio.

Usage:
  python voice_pipeline.py --script line.txt --audio take.mp3 --out build/line01
  python voice_pipeline.py --script line.txt --voice-id <id> --out build/line01   # generate with ElevenLabs

Exit code 0 = take passed QA, 2 = bad take (don't ship), 1 = error.

Env / paths:
  RHUBARB        path to rhubarb binary (default: rhubarb on PATH)
  WHISPER_DIR    sherpa-onnx Whisper export, e.g. sherpa-onnx-whisper-small.en from
                 https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  ELEVENLABS_API_KEY  only needed with --voice-id
"""
import argparse, base64, json, os, re, subprocess, sys
import numpy as np
import soundfile as sf

SR = 16000
MAX_CHUNK = 28.0  # Whisper sees 30 s windows


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


# ---------- 1. generate (optional) ----------
def elevenlabs_tts(text, voice_id, out_mp3):
    """Generate the line and keep ElevenLabs' character alignment for word timings."""
    import urllib.request
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
        data=json.dumps({"text": text, "model_id": os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")}).encode(),
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    open(out_mp3, "wb").write(base64.b64decode(data["audio_base64"]))
    return alignment_to_words(data.get("alignment") or {})


def alignment_to_words(al):
    chars, starts, ends = al.get("characters", []), al.get("character_start_times_seconds", []), al.get("character_end_times_seconds", [])
    words, cur = [], None
    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            if cur: words.append(cur); cur = None
            continue
        if cur is None:
            cur = {"word": ch, "start": s, "end": e}
        else:
            cur["word"] += ch; cur["end"] = e
    if cur: words.append(cur)
    return words


# ---------- 2. Whisper transcript ----------
def split_on_silence(wav, dur):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", wav, "-af", "silencedetect=noise=-35dB:d=0.2", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    mids = []
    for m in re.finditer(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", out):
        end, d = float(m.group(1)), float(m.group(2))
        mids.append(end - d / 2)
    cuts, last = [0.0], 0.0
    for i, m in enumerate(mids):
        nxt = mids[i + 1] if i + 1 < len(mids) else dur
        if nxt - last > MAX_CHUNK and m > last:
            cuts.append(m); last = m
    while dur - cuts[-1] > 30:  # no usable pause: hard split
        cuts.append(cuts[-1] + MAX_CHUNK)
    return cuts + [dur]


def whisper_transcribe(wav, whisper_dir):
    import sherpa_onnx
    name = os.path.basename(whisper_dir.rstrip("/")).replace("sherpa-onnx-whisper-", "")
    p = os.path.join(whisper_dir, name + "-")
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=p + "encoder.int8.onnx", decoder=p + "decoder.int8.onnx", tokens=p + "tokens.txt",
        language="en", task="transcribe", num_threads=os.cpu_count() or 4)
    audio, sr = sf.read(wav, dtype="float32")
    cuts = split_on_silence(wav, len(audio) / sr)
    texts = []
    for a, b in zip(cuts, cuts[1:]):
        st = rec.create_stream()
        st.accept_waveform(sr, audio[int(a * sr):int(b * sr)])
        rec.decode_stream(st)
        texts.append(st.result.text.strip())
    return " ".join(t for t in texts if t)


# ---------- 3. QA: compare transcript to script ----------
NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def norm_words(text):
    text = text.lower().replace("’", "'").replace("-", " ")
    words = re.findall(r"[a-z0-9']+", text)
    return [NUM.get(w, w) for w in words]


# small words Whisper often drops or swaps; errors on these don't count toward --max-content-errors
FILLER = set("a an the and or but so to of in on at for is it its it's i i'm um uh like just that this".split())


def content_errors(diffs):
    return [d for d in diffs if (d.get("script") or d.get("heard")) not in FILLER]


def align(ref, hyp):
    """Word-level Levenshtein; returns (errors, list of diffs)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, diffs = n, m, []
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                diffs.append({"type": "substituted", "script": ref[i - 1], "heard": hyp[j - 1], "at_word": int(i - 1)})
            i, j = i - 1, j - 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            diffs.append({"type": "missing", "script": ref[i - 1], "at_word": int(i - 1)}); i -= 1
        else:
            diffs.append({"type": "extra", "heard": hyp[j - 1], "at_word": int(i)}); j -= 1
    return int(D[n, m]), diffs[::-1]


# ---------- 4. Rhubarb mouth cues ----------
def rhubarb(wav, script_text, workdir):
    dialog = os.path.join(workdir, "dialog.txt")
    open(dialog, "w").write(script_text)
    out = os.path.join(workdir, "rhubarb.json")
    sh([os.environ.get("RHUBARB", "rhubarb"), "-q", "-f", "json", "--extendedShapes", "GHX",
        "--dialogFile", dialog, "-o", out, wav])
    return json.load(open(out))["mouthCues"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="text file with the exact line")
    ap.add_argument("--audio", help="existing take (mp3/wav)")
    ap.add_argument("--voice-id", help="generate the take with ElevenLabs instead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wer", type=float, default=0.08, help="fail the take above this word error rate")
    ap.add_argument("--max-content-errors", type=int, default=2,
                    help="fail if more than this many non-filler words differ (catches flubs a low WER hides)")
    ap.add_argument("--whisper-dir", default=os.environ.get("WHISPER_DIR"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    script_text = open(a.script).read().strip()

    words = None
    src = a.audio
    if a.voice_id:
        src = os.path.join(a.out, "take.mp3")
        words = elevenlabs_tts(script_text, a.voice_id, src)
    if not src:
        sys.exit("need --audio or --voice-id")

    wav = os.path.join(a.out, "take.wav")
    sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(SR), wav])
    dur = sf.info(wav).duration

    transcript = whisper_transcribe(wav, a.whisper_dir)
    ref, hyp = norm_words(script_text), norm_words(transcript)
    errors, diffs = align(ref, hyp)
    wer = errors / max(1, len(ref))
    serious = content_errors(diffs)
    passed = bool(wer <= a.max_wer and len(serious) <= a.max_content_errors)

    cues = rhubarb(wav, script_text, a.out)
    timeline = {
        "audio": os.path.basename(src), "duration": round(dur, 3),
        "script": script_text, "transcript": transcript,
        "qa": {"pass": passed, "wer": round(wer, 4), "maxWer": a.max_wer, "contentErrors": len(serious), "maxContentErrors": a.max_content_errors, "scriptWords": len(ref), "diffs": diffs},
        "words": words,  # from ElevenLabs alignment when generated; null for imported takes
        "mouthCues": cues,  # Rhubarb shapes: A M/B/P, B teeth, C open, D wide, E round, F pucker, G F/V, H L, X rest
    }
    json.dump(timeline, open(os.path.join(a.out, "timeline.json"), "w"), indent=1)
    with open(os.path.join(a.out, "mouth.tsv"), "w") as f:  # animate.py / Rhubarb TSV format
        for c in cues:
            f.write(f"{c['start']:.2f}\t{c['value']}\n")
        if cues:
            f.write(f"{cues[-1]['end']:.2f}\tX\n")

    print(f"{'PASS' if passed else 'FAIL — bad take, do not ship'}  WER {wer:.1%} ({errors} word errors / {len(ref)}, {len(serious)} meaningful)")
    for d in diffs[:20]:
        if d["type"] == "substituted":
            print(f"  word {d['at_word']}: script '{d['script']}' → heard '{d['heard']}'")
        elif d["type"] == "missing":
            print(f"  word {d['at_word']}: missing '{d['script']}'")
        else:
            print(f"  word {d['at_word']}: extra '{d['heard']}'")
    print(f"{len(cues)} mouth cues → {a.out}/timeline.json")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
