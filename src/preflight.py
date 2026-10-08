#!/usr/bin/env python3
"""Pre-flight checks that do NOT need the model weights.

Verifies every manifest row resolves, probes each video with ffprobe, and tokenizes
the real prompt + media to get the exact input length. Run this on a login node or at
the top of a job before paying for a 70GB model load.

  python src/preflight.py --manifest manifest.csv
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTEXT_LIMIT = 32768
WARN_FRACTION = 0.8


def probe(video: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    duration = float(info.get("format", {}).get("duration", 0.0))
    return duration, has_video, has_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--prompt", default="prompts/score_v1.txt")
    ap.add_argument("--tokenize", action="store_true",
                    help="also load the processor and count exact input tokens")
    args = ap.parse_args()

    import csv
    with open(ROOT / args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    problems = []
    totals = []
    if args.tokenize:
        print(f"{'question':10} {'dur(s)':>7} {'A/V':>5} {'total':>7} {'video':>7} "
              f"{'audio':>7} {'text':>6} {'tok/s':>6}  metric")
        print("-" * 96)
    else:
        print(f"{'question':10} {'dur(s)':>7} {'A/V':>5}  metric")
        print("-" * 46)

    processor = None
    if args.tokenize:
        import omni
        from transformers import Qwen3OmniMoeProcessor
        processor = Qwen3OmniMoeProcessor.from_pretrained(omni.MODEL_ID)

    for row in rows:
        qid = row["question_id"]
        video = ROOT / row["video"]
        rubric = ROOT / row["rubric"]

        if not rubric.exists():
            problems.append(f"{qid}: missing rubric {rubric}")
        if not video.exists():
            problems.append(f"{qid}: missing video {video}")
            print(f"{qid:10} {'-':>7} {'-':>5}  MISSING VIDEO")
            continue

        duration, has_v, has_a = probe(video)
        if not has_a:
            problems.append(f"{qid}: no audio stream — the speech channel is the signal")
        if not has_v:
            problems.append(f"{qid}: no video stream")

        av = f"{'V' if has_v else '-'}{'A' if has_a else '-'}"

        if processor is None:
            print(f"{qid:10} {duration:7.1f} {av:>5}  {row['metric']}")
            continue

        import omni
        from score_run import render_prompt
        prompt, _ = render_prompt(
            (ROOT / args.prompt).read_text(encoding="utf-8"),
            {**row, "max_score": int(row.get("max_score") or 5)},
        )
        conv, use_aiv = omni.build_conversation("video", prompt, video, None)
        inputs = omni.prepare_inputs(processor, conv, use_aiv, "cpu", None)
        bd = omni.token_breakdown(processor, inputs)
        totals.append((qid, duration, bd))

        def cell(v):
            return "?" if v is None else f"{v}"

        rate = bd["total"] / duration if duration else 0
        print(f"{qid:10} {duration:7.1f} {av:>5} {bd['total']:7} "
              f"{cell(bd['video']):>7} {cell(bd['audio']):>7} "
              f"{bd['text_and_control']:6} {rate:6.0f}  {row['metric']}")

        if bd["total"] > CONTEXT_LIMIT * WARN_FRACTION:
            problems.append(
                f"{qid}: {bd['total']} input tokens is over {WARN_FRACTION:.0%} of "
                f"{CONTEXT_LIMIT} — lower FPS or VIDEO_MAX_PIXELS"
            )
        for w in bd.get("warnings", []):
            problems.append(f"{qid}: {w} (per-modality split unreliable)")

    if totals:
        print()
        print("=== token budget ===")
        grand = sum(bd["total"] for _, _, bd in totals)
        vids = [bd["video"] for _, _, bd in totals if bd["video"] is not None]
        auds = [bd["audio"] for _, _, bd in totals if bd["audio"] is not None]
        print(f"  largest prompt:  {max(bd['total'] for _, _, bd in totals)} tokens "
              f"({max(bd['total'] for _, _, bd in totals) / CONTEXT_LIMIT:.0%} of "
              f"{CONTEXT_LIMIT})")
        print(f"  all {len(totals)} prompts: {grand} tokens total")
        if vids:
            print(f"  video: {sum(vids)} tokens ({sum(vids)/grand:.0%} of input), "
                  f"mean {statistics.mean(vids):.0f}/clip")
        if auds:
            print(f"  audio: {sum(auds)} tokens ({sum(auds)/grand:.0%} of input), "
                  f"mean {statistics.mean(auds):.0f}/clip")
        secs = sum(d for _, d, _ in totals)
        if secs:
            if vids:
                print(f"  video rate: {sum(vids)/secs:.1f} tok/sec of footage")
            if auds:
                print(f"  audio rate: {sum(auds)/secs:.1f} tok/sec of footage")
        shapes = totals[0][2].get("shapes")
        if shapes:
            print(f"  shapes (first clip): {shapes}")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("All manifest rows resolve.")


if __name__ == "__main__":
    main()
