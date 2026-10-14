# Talking character animation

`talking_animation.mp4` is a 1080x1920 (9:16) 30fps video, 1:13 long, with the ElevenLabs voice track.

- **Lip sync:** [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) v1.14 → `mouth.tsv` (shapes A–H, X). Each shape drives a jaw-drop warp plus a painted mouth interior (teeth/tongue).
- **Scenes:** cuts between `char1.png` (blue studio) and `char2.png` (warm workshop) on speech pauses, with wide, medium and close shots, slow push-ins, and a flash on each scene change.
- **Life:** random blinks, breathing, and a sway/bob that follows loudness.

Rebuild:
```sh
rhubarb -f tsv --extendedShapes GHX voice.wav -o mouth.tsv
python animate.py <dir-with-pngs-and-mouth.tsv> <audio.mp3> out.mp4
```
Needs `numpy`, `opencv-python-headless`, `pillow`, and `ffmpeg`. The script also expects `voice.wav` (16 kHz mono) in that dir.
