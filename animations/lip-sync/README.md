# Talking character animation

`talking_animation.mp4` is an Instagram Reel-size 1080x1920 (9:16) 30fps video, 1:13 long, with the ElevenLabs voice track.

- **Face tracking:** MediaPipe Face Mesh (`detect_face.py` → `landmarks.json`) finds exact lip corners, lip centre, chin and eyelid outlines on both looks.
- **Lip sync:** [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) v1.14 → `mouth.tsv` (shapes A–H, X). Each shape drives a jaw-drop warp plus a painted mouth interior (teeth/tongue).
- **Scenes:** cuts between `char1.png` (blue studio) and `char2.png` (warm workshop) on speech pauses, with wide, medium and close shots, slow push-ins, and a flash on each scene change.
- **Life:** eyelid-shaped blinks (half → closed → half), breathing, and a sway/bob that follows loudness.

Rebuild:
```sh
python detect_face.py <dir>   # needs mediapipe + face_landmarker.task
rhubarb -f tsv --extendedShapes GHX voice.wav -o mouth.tsv
python animate.py <dir-with-pngs-and-mouth.tsv> <audio.mp3> out.mp4
```
Needs `numpy`, `opencv-python-headless`, `pillow`, and `ffmpeg`. The script also expects `voice.wav` (16 kHz mono) in that dir.

## Voice-first pipeline (`voice_pipeline.py`)

The voice take is the master timeline. The pipeline outputs mouth-shape **data**, not video frames, so any rig can play it in sync with the audio:

1. **Generate** the line with ElevenLabs (`--voice-id`, needs `ELEVENLABS_API_KEY`). Its character timestamps become word timings. Or import an existing take with `--audio`.
2. **Whisper QA gate:** offline Whisper small.en (sherpa-onnx export from the [k2-fsa GitHub releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models), so no Hugging Face access needed) transcribes the take, which is compared word by word to the script.
   A take **fails** (exit code 2) if the word error rate is over `--max-wer` (default 8%) **or** more than `--max-content-errors` (default 2) non-filler words differ. The second rule catches a flubbed phrase that a long take's low error rate would hide.
3. **Rhubarb** gets the script as a `--dialogFile` hint and produces mouth cues (A–H, X).
4. **Output:** `timeline.json` (script, transcript, QA verdict and diffs, word timings, mouth cues in Rhubarb's JSON cue format) plus `mouth.tsv` for `animate.py`.

```sh
pip install sherpa-onnx soundfile numpy
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2 && tar xjf sherpa-onnx-whisper-small.en.tar.bz2
export WHISPER_DIR=$PWD/sherpa-onnx-whisper-small.en RHUBARB=/path/to/rhubarb
python voice_pipeline.py --script line.txt --audio take.mp3 --out build/line01 || echo "bad take, re-generate"
```
