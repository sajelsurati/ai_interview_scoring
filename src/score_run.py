#!/usr/bin/env python3
"""Score interview answers with Qwen3-Omni.

One fresh context per (video, mode, run). Writes one JSON per call so a preempted
SLURM job can be requeued and will skip whatever already finished.

  python src/score_run.py --manifest manifest.csv --modes video audio text --runs 3
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import omni

ROOT = Path(__file__).resolve().parent.parent


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["weight"] = float(row.get("weight") or 1)
        row["max_score"] = int(row.get("max_score") or 5)
    return rows


def render_prompt(template: str, row: dict):
    """Returns (prompt, rubric_text). The rubric text comes back so callers can hash
    it: rubric files are edited constantly and are not under version control, so a
    score is only interpretable alongside the text that produced it."""
    rubric = (ROOT / row["rubric"]).read_text(encoding="utf-8").strip()
    prompt = template.format(
        question_text=row["question_text"],
        question_id=row["question_id"],
        metric=row["metric"],
        max_score=row["max_score"],
        rubric=rubric,
    )
    return prompt, rubric


def get_transcript(model, processor, row, video: Path, cache_dir: Path) -> str:
    """Transcribe with the same model, so the text-only arm needs no extra dependency.
    Cached: the transcript is an input to the ablation, not part of what we vary."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{row['question_id']}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    wav = omni.extract_audio(video, cache_dir.parent / "audio")
    conversation, use_aiv = omni.build_conversation(
        "audio", omni.TRANSCRIBE_PROMPT, wav, None
    )
    text, _ = omni.generate(
        model, processor, conversation, use_aiv, max_new_tokens=1024, temperature=0.0
    )
    cached.write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--prompt", default="prompts/score_v1.txt")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--modes", nargs="+", default=["video"],
                    choices=["video", "audio", "text"],
                    help="ablation arms; 'text' uses a model-generated transcript")
    ap.add_argument("--runs", type=int, default=1,
                    help="total passes per cell; run 0 is greedy, runs 1+ are sampled")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="temperature for runs 1+ (run 0 is always greedy)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-stale", action="store_true",
                    help="keep existing results even if the rubric or prompt text has "
                         "changed since they were written (default is to re-run them)")
    args = ap.parse_args()

    manifest = load_manifest(ROOT / args.manifest)
    template = (ROOT / args.prompt).read_text(encoding="utf-8")
    prompt_tag = Path(args.prompt).stem
    prompt_sha = sha12(template)

    # Rubric text is hashed per row so an edited rubric invalidates its old scores.
    for row in manifest:
        _, rubric_text = render_prompt(template, row)
        row["_rubric_sha"] = sha12(rubric_text)

    # Plan the whole grid first, so we can report how much is already done.
    cells = [
        (row, mode, run)
        for row in manifest
        for mode in args.modes
        for run in range(args.runs)
    ]
    out_root = ROOT / args.out

    def out_path(row, mode, run):
        return (out_root / row["interview_id"] / prompt_tag /
                f"{row['question_id']}__{mode}__r{run}.json")

    def is_current(row, mode, run):
        """A finished cell counts as done only if it was produced by the rubric and
        prompt text now on disk. Otherwise `runs/` would quietly mix scores from
        different rubric versions, and the report would average across them."""
        path = out_path(row, mode, run)
        if not path.exists():
            return False
        if args.keep_stale:
            return True
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return (prev.get("rubric_sha") == row["_rubric_sha"]
                and prev.get("prompt_sha") == prompt_sha)

    todo = [c for c in cells if args.overwrite or not is_current(*c)]
    stale = sum(1 for c in cells if out_path(*c).exists() and c in todo)
    print(f"[plan] {len(cells)} cells, {len(cells) - len(todo)} already current, "
          f"{len(todo)} to run" + (f" ({stale} stale, rubric/prompt changed)"
                                   if stale else ""), flush=True)
    if not todo:
        return

    print(f"[load] {omni.MODEL_ID}", flush=True)
    t0 = time.time()
    model, processor = omni.load_model()
    print(f"[load] ready in {time.time() - t0:.1f}s", flush=True)

    for i, (row, mode, run) in enumerate(todo, 1):
        dest = out_path(row, mode, run)
        dest.parent.mkdir(parents=True, exist_ok=True)
        video = ROOT / row["video"]
        if not video.exists():
            print(f"[skip] missing video: {video}", flush=True)
            continue

        cache_dir = out_root / row["interview_id"] / "transcripts"
        prompt, _ = render_prompt(template, row)

        media, transcript = None, None
        if mode == "video":
            media = video
        elif mode == "audio":
            media = omni.extract_audio(video, out_root / row["interview_id"] / "audio")
        else:
            transcript = get_transcript(model, processor, row, video, cache_dir)

        conversation, use_aiv = omni.build_conversation(mode, prompt, media, transcript)
        temperature = 0.0 if run == 0 else args.temperature

        t1 = time.time()
        raw, n_input = omni.generate(
            model, processor, conversation, use_aiv,
            max_new_tokens=args.max_new_tokens, temperature=temperature, seed=run,
        )
        elapsed = time.time() - t1
        parsed, err = omni.parse_score_json(raw)

        record = {
            "interview_id": row["interview_id"],
            "question_id": row["question_id"],
            "metric": row["metric"],
            "mode": mode,
            "run": run,
            "prompt_variant": prompt_tag,
            "prompt_sha": prompt_sha,
            "rubric_sha": row["_rubric_sha"],
            "temperature": temperature,
            "model": omni.MODEL_ID,
            "input_tokens": n_input,
            "seconds": round(elapsed, 2),
            "weight": row["weight"],
            "max_score": row["max_score"],
            "score": parsed.get("score") if parsed else None,
            "parsed": parsed,
            "parse_error": err,
            "raw": raw,
        }
        dest.write_text(json.dumps(record, indent=2), encoding="utf-8")

        flag = "" if parsed else f"  PARSE-FAIL({err})"
        print(f"[{i}/{len(todo)}] {row['interview_id']}/{row['question_id']} "
              f"{mode} r{run} -> score={record['score']} "
              f"({n_input} tok, {elapsed:.1f}s){flag}", flush=True)


if __name__ == "__main__":
    main()
