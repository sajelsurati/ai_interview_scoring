#!/usr/bin/env python3
"""Pre-flight checks that do NOT need the model weights.

Verifies every manifest row resolves, probes each video with ffprobe, and tokenizes
the real prompt + media to get the exact input length. Run this on a login node or at
the top of a job before paying for a 70GB model load.

  python src/preflight.py --manifest manifest.csv
"""

import argparse
import json
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
    print(f"{'question':10} {'dur(s)':>7} {'A/V':>5} {'tokens':>8}  rubric")
    print("-" * 68)

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
            print(f"{qid:10} {'-':>7} {'-':>5} {'-':>8}  MISSING VIDEO")
            continue

        duration, has_v, has_a = probe(video)
        if not has_a:
            problems.append(f"{qid}: no audio stream — the speech channel is the signal")
        if not has_v:
            problems.append(f"{qid}: no video stream")

        n_tok = "-"
        if processor is not None:
            import omni
            from score_run import render_prompt
            prompt, _ = render_prompt(
                (ROOT / args.prompt).read_text(encoding="utf-8"),
                {**row, "max_score": int(row.get("max_score") or 5)},
            )
            conv, use_aiv = omni.build_conversation("video", prompt, video, None)
            inputs = omni.prepare_inputs(processor, conv, use_aiv, "cpu", None)
            n_tok = int(inputs["input_ids"].shape[1])
            if n_tok > CONTEXT_LIMIT * WARN_FRACTION:
                problems.append(
                    f"{qid}: {n_tok} input tokens is over {WARN_FRACTION:.0%} of "
                    f"{CONTEXT_LIMIT} — lower FPS or max_pixels"
                )

        av = f"{'V' if has_v else '-'}{'A' if has_a else '-'}"
        print(f"{qid:10} {duration:7.1f} {av:>5} {str(n_tok):>8}  {row['metric']}")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("All manifest rows resolve.")


if __name__ == "__main__":
    main()
