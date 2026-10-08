#!/usr/bin/env python3
"""Aggregate run JSONs into per-question scores, interview totals, and the
reliability / ablation statistics that make this a test rather than a demo.

The total is computed here in Python. The model is never asked to add up its own
scores -- that is deterministic arithmetic and asking an LLM for it only adds a
failure mode.

  python src/aggregate.py --runs runs --out reports
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODE_ORDER = ["video", "audio", "text"]


def load_records(runs_dir: Path):
    records = []
    for path in sorted(runs_dir.rglob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"[warn] unreadable run file: {path}")
    return records


def pairwise_agreement(scores):
    """Returns (exact, within_one) as fractions over all pairs, or (None, None)."""
    vals = [s for s in scores if s is not None]
    if len(vals) < 2:
        return None, None
    pairs = list(combinations(vals, 2))
    exact = sum(1 for a, b in pairs if a == b) / len(pairs)
    within = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
    return exact, within


def summarize_cell(recs):
    """One (interview, variant, mode, question) cell across its runs."""
    greedy = [r for r in recs if r["run"] == 0]
    sampled = [r for r in recs if r["run"] != 0]
    all_scores = [r["score"] for r in recs]
    present = [s for s in all_scores if s is not None]

    if greedy and greedy[0]["score"] is not None:
        point = greedy[0]["score"]
        point_src = "greedy"
    elif present:
        point = statistics.median(present)
        point_src = "median"
    else:
        point, point_src = None, "none"

    exact, within = pairwise_agreement([r["score"] for r in sampled] or all_scores)
    return {
        "point": point,
        "point_source": point_src,
        "n_runs": len(recs),
        "n_null": sum(1 for s in all_scores if s is None),
        "n_parse_fail": sum(1 for r in recs if r.get("parse_error")),
        "scores": all_scores,
        "spread": (max(present) - min(present)) if len(present) >= 2 else None,
        "agree_exact": exact,
        "agree_within_one": within,
        "metric": recs[0]["metric"],
        "weight": recs[0]["weight"],
        "max_score": recs[0]["max_score"],
        "mean_seconds": statistics.mean(r["seconds"] for r in recs),
        "mean_input_tokens": statistics.mean(r["input_tokens"] for r in recs),
    }


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    records = load_records(ROOT / args.runs)
    if not records:
        raise SystemExit(f"no run JSONs under {ROOT / args.runs}")

    cells = defaultdict(list)
    for r in records:
        key = (r["interview_id"], r["prompt_variant"], r["mode"], r["question_id"])
        cells[key].append(r)
    summary = {k: summarize_cell(v) for k, v in cells.items()}

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- long-form CSV -----------------------------------------------------
    csv_path = out_dir / "scores_long.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["interview_id", "prompt_variant", "mode", "question_id", "metric",
                    "point_score", "point_source", "max_score", "weight", "n_runs",
                    "all_scores", "spread", "agree_exact", "agree_within_one",
                    "n_null", "n_parse_fail", "mean_input_tokens", "mean_seconds"])
        for (iid, var, mode, qid), s in sorted(summary.items()):
            w.writerow([iid, var, mode, qid, s["metric"], fmt(s["point"]),
                        s["point_source"], s["max_score"], s["weight"], s["n_runs"],
                        "|".join("null" if x is None else str(x) for x in s["scores"]),
                        fmt(s["spread"]), fmt(s["agree_exact"]),
                        fmt(s["agree_within_one"]), s["n_null"], s["n_parse_fail"],
                        fmt(s["mean_input_tokens"], 0), fmt(s["mean_seconds"], 1)])

    # ---- markdown report ---------------------------------------------------
    lines = ["# Qwen3-Omni interview scoring report", ""]
    interviews = sorted({k[0] for k in summary})
    variants = sorted({k[1] for k in summary})
    modes = [m for m in MODE_ORDER if m in {k[2] for k in summary}]

    n_fail = sum(s["n_parse_fail"] for s in summary.values())
    n_total = sum(s["n_runs"] for s in summary.values())
    lines += [f"- model calls: **{n_total}**",
              f"- parse failures: **{n_fail}** ({n_fail / n_total:.1%})",
              f"- modes: {', '.join(modes)}",
              f"- prompt variants: {', '.join(variants)}", ""]

    # Rubric files are edited freely and are not version controlled. If runs/ holds
    # scores produced by more than one version of a rubric, averaging them is
    # meaningless -- say so loudly rather than quietly reporting a blended number.
    rubric_versions = defaultdict(set)
    for r in records:
        if r.get("rubric_sha"):
            rubric_versions[r["metric"]].add(r["rubric_sha"])
    mixed = {m: v for m, v in rubric_versions.items() if len(v) > 1}
    if mixed:
        lines += ["> **Warning: mixed rubric versions.** These metrics have scores from "
                  "more than one version of their rubric text, so their numbers are not "
                  "comparable:", ""]
        for m, shas in sorted(mixed.items()):
            lines.append(f"> - `{m}`: {len(shas)} versions ({', '.join(sorted(shas))})")
        lines += ["",
                  "> Re-run with `--overwrite`, or delete the affected files in `runs/`.",
                  ""]
    if any(not r.get("rubric_sha") for r in records):
        lines += ["> Note: some runs predate rubric-hash recording and cannot be "
                  "checked for staleness.", ""]

    for iid in interviews:
        lines += [f"## Interview `{iid}`", ""]
        for var in variants:
            qids = sorted({k[3] for k in summary if k[0] == iid and k[1] == var})
            if not qids:
                continue
            lines += [f"### Scores by modality — prompt `{var}`", "",
                      "| question | metric | " + " | ".join(modes) + " |",
                      "|---|---|" + "---|" * len(modes)]
            for qid in qids:
                cellrow = []
                metric = ""
                for mode in modes:
                    s = summary.get((iid, var, mode, qid))
                    if s is None:
                        cellrow.append("-")
                        continue
                    metric = s["metric"]
                    stab = ""
                    if s["agree_exact"] is not None:
                        stab = f" <sub>({s['agree_exact']:.0%} agree)</sub>"
                    cellrow.append(f"{fmt(s['point'], 1)}{stab}")
                lines.append(f"| {qid} | {metric} | " + " | ".join(cellrow) + " |")

            # Totals: sum of weighted point scores. Computed, not asked for.
            lines += ["", "| modality | total | max | pct | scored questions |",
                      "|---|---|---|---|---|"]
            for mode in modes:
                got = maxt = 0.0
                n_scored = 0
                for qid in qids:
                    s = summary.get((iid, var, mode, qid))
                    if s is None or s["point"] is None:
                        continue
                    got += s["point"] * s["weight"]
                    maxt += s["max_score"] * s["weight"]
                    n_scored += 1
                pct = f"{got / maxt:.1%}" if maxt else "-"
                lines.append(f"| {mode} | {got:.1f} | {maxt:.1f} | {pct} | "
                             f"{n_scored}/{len(qids)} |")
            lines.append("")

            # ---- discrimination: does the rubric separate the answers? ----
            lines += ["#### Discrimination (spread of scores across questions)", ""]
            for mode in modes:
                pts = [summary[(iid, var, mode, q)]["point"] for q in qids
                       if (iid, var, mode, q) in summary
                       and summary[(iid, var, mode, q)]["point"] is not None]
                if len(pts) >= 2:
                    sd = statistics.pstdev(pts)
                    note = "  <-- no variance; rubric is not discriminating" if sd < 0.5 else ""
                    lines.append(f"- {mode}: sd={sd:.2f}, range={min(pts):.0f}-{max(pts):.0f}{note}")
                else:
                    lines.append(f"- {mode}: too few scored questions")
            lines.append("")

            # ---- ablation: what do the non-text channels change? ----------
            if "video" in modes and "text" in modes:
                lines += ["#### Ablation: full recording vs transcript-only", "",
                          "| question | metric | video | text | delta |",
                          "|---|---|---|---|---|"]
                deltas = []
                for qid in qids:
                    sv = summary.get((iid, var, "video", qid))
                    st = summary.get((iid, var, "text", qid))
                    if not sv or not st:
                        continue
                    d = (None if sv["point"] is None or st["point"] is None
                         else sv["point"] - st["point"])
                    if d is not None:
                        deltas.append(abs(d))
                    lines.append(f"| {qid} | {sv['metric']} | {fmt(sv['point'], 1)} | "
                                 f"{fmt(st['point'], 1)} | {fmt(d, 1)} |")
                if deltas:
                    mad = statistics.mean(deltas)
                    verdict = ("transcript-only reproduces the full-recording scores; "
                               "the audio and video are not changing the judgment"
                               if mad < 0.5 else
                               "the audio and video are moving the scores, so they are "
                               "doing real work")
                    lines += ["", f"Mean |delta| = **{mad:.2f}** — {verdict}.", ""]

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
