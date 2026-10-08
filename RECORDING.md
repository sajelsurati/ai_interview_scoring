# Recording checklist

Phone recording is fine for this test. The defaults are not — fix them first.

## iPhone settings (Settings → Camera)

| Setting | Set to | Why |
|---|---|---|
| Formats | **Most Compatible** | records H.264 + AAC; the HEVC default fails or drops frames in some `decord`/`av` builds |
| Record Video | **1080p / 30 fps** | 4K is wasted — it gets downscaled to 480p |
| Record Video → HDR Video | **Off** | Dolby Vision tonemaps badly through transcode |
| Record Video → Auto FPS | **Off** | otherwise drops to 24 fps in dim light, producing variable frame rate |

4K60, ProRes, and HDR force HEVC regardless of the Formats setting. Leave them off.

## Setup

- **Landscape** orientation. Portrait wastes tokens on wall and crops out gesture.
- Head and shoulders in frame, phone 60–90 cm away, lens roughly at eye level.
- Stable mount. Handheld wobble reads as nervous delivery.
- Quiet room. Reverb degrades both transcription and any judgment about delivery.
- One file per answer. Start recording after the question is asked, or trim the head off
  — if the interviewer is audible, the model hears and scores it.

## Keep conditions constant

Identical framing, distance, lighting, and room across **all clips and all candidates.**

Any metric the model reads delivery cues for — `sociability` most obviously — can pick up
recording quality instead of the candidate. Nothing in the rubrics tells the model to
ignore lighting or camera angle. If one candidate is filmed in worse light and scores
lower on sociability, you cannot separate the two after the fact.

## Transfer

AirDrop, USB via Finder/Image Capture, or the Files app — all preserve the original.
Never route clips through iMessage, WhatsApp, or Google Photos "storage saver"; they
re-encode hard.

This project lives in iCloud Drive. With "Optimize Mac Storage" on, files can remain
placeholders and `rsync` will stall or copy nothing. Materialize them first:

```bash
brctl download videos/          # or just open each file once
```

## Then

```bash
bash tools/normalize_videos.sh raw/demo01 videos/demo01   # H.264, 480p, CFR, mono
# trim each clip to the candidate's answer only, e.g.
#   ffmpeg -i videos/demo01/q1.mp4 -ss 00:00:03 -to 00:00:48 -c copy videos/demo01/q1t.mp4

# Use the data transfer node, not the login node, for bulk copies.
# Off campus, connect to the NYU VPN first.
rsync -avP videos/ <NetID>@dtn.torch.hpc.nyu.edu:/scratch/<NetID>/ai_interview_scoring/videos/

# then on Torch (ssh <NetID>@login.torch.hpc.nyu.edu):
python src/preflight.py --manifest manifest.csv
```
