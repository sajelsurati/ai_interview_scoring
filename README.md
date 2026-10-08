# Interview scoring test — Qwen3-Omni-30B-A3B-Instruct

A short, honest test of whether a local omni model can score interview answers against
a fixed rubric. N≤5 answers per interview, one metric per answer, <1 min per clip,
2×80GB GPUs, BF16.

## What the test actually measures

There are no human scores for these videos, so **this cannot report accuracy**. It
reports three things that are informative anyway:

1. **Self-consistency** — 3 passes per answer (1 greedy + 2 sampled). If the model
   can't agree with itself, agreement with a human is moot.
2. **Discrimination** — the spread of scores across the 5 questions. If everything
   comes back a 4, the rubric isn't separating anything and the test has told you
   nothing.
3. **Modality ablation** — the same answer scored from full video, audio only, and a
   transcript. **This is the centerpiece.** If transcript-only reproduces the video
   scores, a 30B omni model isn't earning its cost over Whisper + a text LLM.

Note that the current metric set is all content-based — nothing in it requires seeing or
hearing the candidate. So transcript-only *should* roughly reproduce the full-recording
scores, and a small ablation delta is the correct answer here, not a warning sign. Read
the ablation as "how much does the extra input perturb a content judgment," and add a
delivery-dependent metric if you want it to test anything stronger.

## Design decisions worth knowing

**The rubrics have no level anchors at all.** Each file names a skill and defines it in a
sentence. Nothing says what a 1 or a 5 looks like. The 1–`max_score` range comes from the
prompt; what the numbers *mean* is left entirely to the model.

This is the deliberate extreme of keeping the scale broad, and it is the cleanest
baseline: the scores are the model's own calibration for each skill, uncontaminated by
anchor wording it could keyword-match against. It costs you two things, both measurable
in the report rather than hidden:

- **Reliability should drop.** Nothing pins the levels, so re-runs have more room to
  disagree. The `(n% agree)` column is where this shows up.
- **The total gets softer.** A 4 on `work_ethic` and a 4 on
  `expertise_in_espresso_drink_preparation` need not represent comparable attainment, so
  summing them is a rougher operation than it looks. The per-question scores are the
  trustworthy output; treat the total as a convenience.

**Nothing in the rubric or prompt partitions the recording.** The model is given the whole
answer — picture, voice, and words together — and asked to rate the skill. It is not told
which aspects of the recording to attend to for which metric.

**Neither prompt says anything about how to use the scale.** No "use the full range", no
warning against rewarding polish. The scores are the model's unprompted behavior, which
is what you want to measure first — a prompt that forbids clustering at 3 can manufacture
spread that isn't judgment, and then the discrimination statistic looks healthy while
measuring the instruction rather than the model.

**`v2` differs from `v1` in wording only.** Same steps, same constraints, same output
schema, different phrasing. So the v1/v2 delta isolates wording sensitivity: if scores
move by more than a point between them, the rubric levels are carrying less of the
judgment than the prompt phrasing is.

**One fresh context per (video, mode, run).** The model process and weights load once;
the context is rebuilt from scratch for every call — system prompt + one rubric + one
video, nothing else. No history, no KV cache carryover. A single rolling conversation
across all five answers would introduce anchoring, order effects, and context growth,
and would make individual questions impossible to retry.

**The total is computed in Python, never by the model.** It is deterministic arithmetic
over the per-question scores; asking an LLM to sum them only adds a failure mode.

**Evidence before score.** The output schema forces timestamped observations and a
justification *before* the integer, so the score is grounded rather than vibed.

**Parse failures are recorded, not rescued.** An unparseable response becomes
`score: null` with the raw text kept. No regex fishes a number out of prose.

**Rubric and prompt text are hashed into every run record.** `runs/` is idempotent, so
editing a rubric would otherwise leave the old scores in place and the report would
silently average across rubric versions. A changed hash counts as a cache miss and the
cell re-runs; `aggregate.py` prints a loud warning if `runs/` still holds mixed versions.
Pass `--keep-stale` to suppress the re-run. This matters because the repo isn't under
version control — if you plan to iterate much, `git init` is worth the thirty seconds.

**The Talker is disabled** (`model.disable_talker()`). We only ever want text out, and
dropping it frees ~10GB of GPU memory.

## Layout

```
requirements.txt          pip deps (no ffmpeg — conda; no flash-attn — see below)
requirements-flashattn.txt  flash-attn alone; needs --no-build-isolation after torch
requirements.lock.txt     written by setup_env.sh after a successful build; preferred
manifest.csv              one row per question: text, metric, rubric path, video, weight
rubrics/<metric>.md       one skill per file: a name and a one-sentence definition
prompts/score_v1.txt      primary scoring prompt
prompts/score_v2.txt      same instructions reworded, for wording-sensitivity testing
src/omni.py               model load, conversation building, generation, JSON parsing
src/score_run.py          driver: manifest x modes x runs, one JSON per cell
src/preflight.py          manifest + ffprobe + exact token count, no weights needed
src/aggregate.py          medians, totals, reliability, discrimination, ablation
tools/normalize_videos.sh ffmpeg normalization to the format the decoders like
slurm/smoke.sbatch        rung 1: does the model load and emit text?
slurm/score.sbatch        the full run
runs/                     one JSON per model call (idempotent; requeue-safe)
reports/                  report.md + scores_long.csv
```

## Setup (NYU Torch)

The python env lives in an Apptainer overlay because conda creates >100k files and
`/scratch` caps you at 1M inodes.

Put your allocation in `.env` once:

```bash
printf 'ACCOUNT=torch_pr_XXXX_XXXXX\n' >> .env
chmod 600 .env
```

Then build the environment as a batch job — overlay + miniforge + torch/transformers
+ ~70 GB of weights:

```bash
./submit.sh slurm/setup.sbatch
tail -f logs/omni-setup-*.out
```

Torch specifics that differ from Greene:

- **`--account` is mandatory**, and `#SBATCH` directives cannot reference shell
  variables — SLURM reads them as literal text before any shell runs. So the sbatch
  files carry no `--account` line, and `./submit.sh` supplies it from `.env` via
  `SBATCH_ACCOUNT`. Submit through the wrapper, not `sbatch` directly.
- **Overlay templates:** `/share/apps/overlay-fs-ext3/` (not Greene's
  `/scratch/work/public/overlay-fs-ext3/`).
- **Base images:** `/share/apps/images/`. Run `ls /share/apps/images/ | grep cuda` and
  take a recent one; `setup_env.sh` prints the list.
- **`--fakeroot` is needed** to mount an overlay `:rw` when building the env. Production
  runs mount `:ro` and don't need it.
- **`/scratch` is flushed after 60 days without access** — the 70GB of weights will
  disappear if the project sits idle.

Pin `transformers` to whatever ref the model card asks for rather than trusting
`latest` — the `Qwen3OmniMoe*` classes are recent and the API has moved.

### Hugging Face token (optional)

`setup_env.sh` reads `.env` from the project root if present:

```bash
printf 'HF_TOKEN=hf_your_token_here\n' > .env
chmod 600 .env
```

`HF_TOKEN` is **optional** — Qwen3-Omni-30B-A3B-Instruct is public and downloads
without one. A token only raises your rate limits, which helps on a ~70 GB pull.

Three things worth knowing:

- **The file must exist on Torch**, not just on your laptop — that's where the download
  runs. `rsync` will carry it over; `ENV_FILE=/path/to/.env` overrides the location.
- **Only `setup_env.sh` uses it.** The scoring jobs run `HF_HUB_OFFLINE=1` against
  already-downloaded weights, so no token ever reaches a scoring job.
- **It's passed via `APPTAINERENV_HF_TOKEN`, not on the command line.**
  `/proc/<pid>/cmdline` is world-readable on a shared node, so a token in the
  `singularity exec` string would be visible to every other user; `/proc/<pid>/environ`
  is owner-only. The setup script never prints the value, only the key names.

`.env` is gitignored. So are `videos/`, `raw/`, `runs/`, and `reports/` — recordings and
run outputs contain identifiable personal data (faces, voices, verbatim transcripts), and
none of it belongs in version control or anywhere you share onward.

### Dependencies

`setup_env.sh` installs from `requirements.txt`, then `requirements-flashattn.txt` in a
second pass. Three packages can't live in one file:

- **`ffmpeg`** is a conda-forge package, not pip. Installed by `setup_env.sh` directly.
- **`flash-attn`** compiles from source and its `setup.py` imports torch, so it needs
  `--no-build-isolation` *and* a torch that is already installed. It also needs
  `MAX_JOBS=4` or the parallel `nvcc` processes will exhaust node RAM. It is **optional**
  — `src/omni.py` tries `flash_attention_2`, logs why it failed, and falls back to
  PyTorch's fused `sdpa` kernel. Same outputs, somewhat slower. A failed build warns
  instead of aborting the whole setup.
- **`torch`** comes from default PyPI, whose wheels bundle CUDA and support H100/H200
  (sm_90). Add an `--index-url` line to `requirements.txt` only if you need a specific
  CUDA build.

Bounds in `requirements.txt` are loose deliberately — hard pins that have never been
resolved against Torch's CUDA image turn a working build into a solver conflict. After a
successful build, `setup_env.sh` writes `requirements.lock.txt` (flash-attn filtered out)
and prefers it on every subsequent run, so rebuilds reproduce the resolution that worked.

Force a backend at any time with `QWEN_OMNI_ATTN=sdpa|eager|flash_attention_2`.

## GPU sizing on Torch

A **single H200 (141GB) holds the BF16 model comfortably** — no multi-GPU setup, no
`device_map` sharding across cards. That's the default in the sbatch scripts.

| GPU | Memory | Verdict |
|---|---|---|
| H200 | 141 GB | best fit; 1 GPU is plenty |
| H100 | 80 GB | works on 1 GPU but tight (~68–78 GB for a video request); 2 is comfortable |
| A100 | 40 / 80 GB | 80GB only, and tight; 40GB needs 2+ |
| L40S | 48 GB | too small for BF16 — would need FP8/AWQ |
| B200 | 180 GB | fits, but needs CUDA 12.8+; check the base image supports sm_100 |

`--gres=gpu:1` doesn't guarantee a GPU *type*. Confirm the feature name for the type you
want before relying on `--constraint`:

```bash
sinfo -o "%20P %10G %8D %12m %f" | sort -u
```

Both sbatch scripts print `nvidia-smi` output first, so if you land on an L40S you'll see
it in the log rather than discovering it as an OOM after a long weight load.

## Video format

MP4, **H.264 + AAC**, short side ≤480px, constant frame rate, mono audio, **one answer
per file, trimmed to the candidate's speech only**, at
`videos/<interview_id>/<question_id>.mp4`.

Recording on a phone is fine — see **[RECORDING.md](RECORDING.md)** for the settings to
change first (iPhone defaults to HEVC and Dolby Vision, both of which break the decoders)
and why recording conditions must stay constant across candidates.

```bash
bash tools/normalize_videos.sh raw/demo01 videos/demo01
```

- H.265/HEVC (the iPhone default) fails or drops frames in some `decord`/`av` builds.
- Every file must have a real audio stream; `preflight.py` fails the run if one doesn't.
- Variable frame rate breaks frame-timestamp math when sampling at 1 fps.
- If the interviewer's question is audible at the head of a clip, **the model will hear
  and score it**. This is the most likely way to quietly corrupt the results.

Token budget is controlled at inference time, not by the file: `FPS=1.0` and
`VIDEO_MAX_PIXELS` in `slurm/score.sbatch`. At 1 fps a 50-second clip is a few thousand
frame tokens plus ~25 audio tokens/sec — comfortable in a 32k window.

## Run it

Climb the ladder; don't discover problems after a two-hour queue wait.

```bash
python src/preflight.py --manifest manifest.csv            # 0. login node, no GPU
./submit.sh slurm/smoke.sbatch                             # 1. model loads, text out
# 2. one short clip, one question — copy manifest.csv to manifest_smoke.csv, keep the
#    header plus one row, point it at a ~10s trim of any answer, then:
#    python src/score_run.py --manifest manifest_smoke.csv --modes video --runs 1
./submit.sh slurm/score.sbatch                             # 3. full grid + report
```

The full grid is 5 questions × 3 modes × 3 runs + 5 prompt-variant calls = 50 calls.
`runs/` is checked before each call, so a preempted job can be requeued and resumes.

## Token accounting

Every call records how many tokens each modality cost. The split is measured, not
estimated: the processor expands one placeholder in the chat template into N copies of a
per-modality pad token, so counting those pad tokens in `input_ids` gives the exact cost
without replicating the patch-merge or audio-pooling arithmetic.

Where it shows up:

- **`src/preflight.py --tokenize`** — per-clip table (total / video / audio / text /
  tokens-per-second) plus a budget summary, before any GPU time is spent. This is where
  you tune `FPS` and `VIDEO_MAX_PIXELS`.
- **`score_run.py` log lines** — `[8421 tok (vid 6912 / aud 1180 / txt 329), 31.4s]`.
- **Each run JSON** — a `tokens` object with per-modality counts, `output` tokens, and
  `shapes` (`video_grid_thw`, mel-frame count, pixel tensor dims).
- **`reports/report.md`** — mean cost per call by mode, and video/audio/text share of
  the full-recording prompt.
- **`reports/scores_long.csv`** — `mean_video_tokens`, `mean_audio_tokens`,
  `mean_text_tokens`, `mean_output_tokens`.

If the pad token ids can't be resolved — the token names have moved between Qwen
releases — the per-modality values come back `null` with a `warnings` entry rather than
a misleading `0`, `preflight` reports it as a problem, and `smoke.sbatch` prints the
resolved ids so you find out on the first job rather than at analysis time. Totals are
always correct regardless, since they're just the sequence length.

## Reading the report

`reports/report.md` gives per-question scores by modality, interview totals, and:

- **`(n% agree)`** — pairwise exact agreement across sampled runs. Low values here mean
  your point estimates are noise.
- **Discrimination sd < 0.5** — flagged explicitly. The rubric is not separating answers.
- **Mean |delta| video vs text < 0.5** — flagged explicitly. The audio and video are not
  changing the scores.

## Known limits

- No ground truth, so no validity claim. Even 5–10 hand-scored answers would turn this
  from a feasibility demo into a calibration test; that's the obvious next step.
- The transcript for the text-only arm is generated by the same model, so that arm tests
  "same model, less input" rather than "Qwen-Omni vs. a dedicated ASR + text LLM".
- Instruct is a non-reasoning variant. `Qwen3-Omni-30B-A3B-Thinking` is worth a second
  arm — rubric application is the kind of deliberative task where it tends to help.
- Scoring real candidates with this would need fairness analysis (accent, appearance,
  and recording-quality effects, which bleed into any metric the model reads delivery
  cues for) well beyond what a feasibility test covers.
