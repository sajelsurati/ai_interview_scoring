"""Shared model loading and generation for Qwen3-Omni interview scoring.

One process loads the model once; every scoring call builds a fresh context
(system prompt + one rubric + one video). Nothing carries over between calls.

torch/transformers are imported lazily inside the functions that need them, so the
pure helpers here (parsing, ffmpeg, conversation building) are importable and testable
without a GPU environment.
"""

import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

MODEL_ID = os.environ.get("QWEN_OMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")

SYSTEM_PROMPT = (
    "You are a careful, calibrated interview assessor. You apply rubrics with judgment, "
    "ground every rating in what you observed, and output only the JSON asked for."
)

TRANSCRIBE_PROMPT = (
    "Transcribe the spoken audio verbatim. Include filler words. Do not summarize, "
    "correct, or comment. Output only the transcript."
)


def load_model(attn: str | None = None):
    """Load weights once. Talker is disabled: we only ever want text out, and
    dropping it frees roughly 10GB of GPU memory.

    Attention backend: flash_attention_2 is preferred, but it compiles from source
    and the build is fragile. Falling back to PyTorch's fused sdpa kernel costs some
    throughput and nothing in accuracy -- better than losing a queued job over a
    failed wheel. Override with QWEN_OMNI_ATTN=sdpa|eager|flash_attention_2.
    """
    from transformers import (Qwen3OmniMoeForConditionalGeneration,
                              Qwen3OmniMoeProcessor)

    requested = attn or os.environ.get("QWEN_OMNI_ATTN")
    candidates = [requested] if requested else ["flash_attention_2", "sdpa"]

    model, last_error = None, None
    for impl in candidates:
        try:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                MODEL_ID,
                dtype="auto",
                device_map="auto",
                attn_implementation=impl,
            )
            print(f"[load] attention backend: {impl}", flush=True)
            break
        except (ImportError, ValueError, RuntimeError) as exc:
            # ImportError: flash-attn not installed. ValueError: transformers rejects
            # the name. RuntimeError: kernel present but unusable on this GPU.
            print(f"[load] {impl} unavailable ({type(exc).__name__}: {exc}); "
                  f"trying next", flush=True)
            last_error = exc
    if model is None:
        raise RuntimeError(
            f"no usable attention backend from {candidates}: {last_error}"
        )

    model.disable_talker()
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
    return model, processor


def extract_audio(video: Path, out_dir: Path) -> Path:
    """Demux a 16kHz mono wav for the audio-only arm. Qwen's audio tower expects 16kHz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{video.stem}.wav"
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-vn", "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )
    return wav


def build_conversation(mode: str, prompt: str, media: Path | None, transcript: str | None):
    """Returns (conversation, use_audio_in_video).

    mode="video"  -> the full recording: frames plus the video's own audio track
    mode="audio"  -> audio track only, no frames
    mode="text"   -> model-generated transcript only, no media at all
    """
    if mode == "video":
        content = [{"type": "video", "video": str(media)},
                   {"type": "text", "text": prompt}]
        use_audio_in_video = True
    elif mode == "audio":
        content = [{"type": "audio", "audio": str(media)},
                   {"type": "text", "text": prompt}]
        use_audio_in_video = False
    elif mode == "text":
        content = [{"type": "text",
                    "text": f"TRANSCRIPT OF THE ANSWER\n{transcript}\n\n{prompt}"}]
        use_audio_in_video = False
    else:
        raise ValueError(f"unknown mode: {mode}")

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return conversation, use_audio_in_video


def _media_kwargs(processor, audios, images, videos):
    """Map media onto the parameter names this processor version actually accepts.

    Qwen3-Omni's processor takes `audio` (singular) but `images` and `videos`
    (plural). Getting it wrong is nasty rather than obvious: transformers only WARNS
    on an unrecognized kwarg and then drops the media, so the chat template still
    inserts <|VIDEO|> placeholders and the failure surfaces much later as
    `StopIteration` inside replace_multimodal_special_tokens. Resolve the names from
    the real signature so a future rename cannot reintroduce that.
    """
    import inspect

    try:
        params = set(inspect.signature(processor.__call__).parameters)
    except (TypeError, ValueError):
        params = set()

    def pick(candidates, value):
        for name in candidates:
            if name in params:
                return {name: value}
        # Signature unavailable or **kwargs-only: use the documented name.
        return {candidates[0]: value}

    kwargs = {}
    kwargs.update(pick(("audio", "audios"), audios))
    kwargs.update(pick(("images", "image"), images))
    kwargs.update(pick(("videos", "video"), videos))
    return kwargs


def prepare_inputs(processor, conversation, use_audio_in_video, device, dtype):
    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text,
        **_media_kwargs(processor, audios, images, videos),
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(device)
    if dtype is not None:
        # BatchFeature.to(dtype) casts only floating-point tensors, so token ids
        # stay integral.
        inputs = inputs.to(dtype)
    return inputs


# --- per-modality token accounting ------------------------------------------
#
# The processor expands one placeholder in the chat template into N copies of a
# per-modality pad token, where N is however many tokens that media actually costs.
# Counting those pad tokens in input_ids therefore gives the exact split, with no
# need to replicate the patch-merge or audio-pooling arithmetic ourselves.
#
# Token *names* and config *attribute* names have both moved between Qwen releases,
# so resolution tries several candidates and reports failure rather than returning a
# silent zero -- a zero that means "not measured" would be worse than no number.

_PAD_TOKEN_NAMES = {
    "audio": ("<|audio_pad|>", "<|AUDIO|>", "<|audio|>"),
    "video": ("<|video_pad|>", "<|VIDEO|>", "<|video|>"),
    "image": ("<|image_pad|>", "<|IMAGE|>", "<|image|>"),
}

_CONFIG_ID_ATTRS = {
    "audio": ("audio_token_index", "audio_token_id"),
    "video": ("video_token_index", "video_token_id"),
    "image": ("image_token_index", "image_token_id"),
}


def resolve_modality_token_ids(processor, model=None):
    """Best-effort map of modality -> pad token id. Returns (ids, warnings)."""
    ids, warnings = {}, []
    tokenizer = getattr(processor, "tokenizer", None)
    unk = getattr(tokenizer, "unk_token_id", None) if tokenizer else None

    configs = []
    if model is not None:
        cfg = getattr(model, "config", None)
        for holder in (cfg,
                       getattr(cfg, "thinker_config", None),
                       getattr(cfg, "text_config", None)):
            if holder is not None:
                configs.append(holder)

    for modality, names in _PAD_TOKEN_NAMES.items():
        token_id = None

        if tokenizer is not None:
            for name in names:
                try:
                    candidate = tokenizer.convert_tokens_to_ids(name)
                except Exception:          # noqa: BLE001 - tokenizer impls vary
                    candidate = None
                if candidate is not None and candidate >= 0 and candidate != unk:
                    token_id = int(candidate)
                    break

        if token_id is None:
            for holder in configs:
                for attr in _CONFIG_ID_ATTRS[modality]:
                    value = getattr(holder, attr, None)
                    if value is not None:
                        token_id = int(value)
                        break
                if token_id is not None:
                    break

        if token_id is None:
            warnings.append(f"{modality}: pad token id unresolved")
        else:
            ids[modality] = token_id

    return ids, warnings


def token_breakdown(processor, inputs, model=None):
    """Split the prompt's token count by modality.

    Returns a dict with per-modality counts, the leftover ("text_and_control":
    the chat template, the rubric, and the bos/eos markers wrapping each media
    span), and the raw media tensor shapes, which are what you tune FPS and
    VIDEO_MAX_PIXELS against.
    """
    input_ids = inputs["input_ids"]
    # .tolist() rather than tensor ops: works for torch/numpy/plain lists alike,
    # and a few thousand ints is free to count in Python.
    flat = input_ids.reshape(-1).tolist() if hasattr(input_ids, "reshape") else list(input_ids)
    total = len(flat)

    ids, warnings = resolve_modality_token_ids(processor, model)
    counts = Counter(flat)
    per_modality = {m: int(counts.get(tid, 0)) for m, tid in ids.items()}

    out = {
        "total": total,
        "audio": per_modality.get("audio"),
        "video": per_modality.get("video"),
        "image": per_modality.get("image"),
    }
    measured = sum(v for v in per_modality.values())
    out["media"] = measured
    out["text_and_control"] = total - measured
    if warnings:
        out["warnings"] = warnings

    # Raw shapes: frame grid and mel-frame count. Useful on their own -- they tell
    # you whether FPS/max_pixels took effect, independent of tokenization.
    shapes = {}
    if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
        shapes["video_grid_thw"] = inputs["video_grid_thw"].tolist()
    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        shapes["image_grid_thw"] = inputs["image_grid_thw"].tolist()
    for key in ("pixel_values_videos", "pixel_values", "input_features"):
        if key in inputs and inputs[key] is not None:
            shapes[f"{key}_shape"] = list(inputs[key].shape)
    if "feature_attention_mask" in inputs and inputs["feature_attention_mask"] is not None:
        shapes["audio_mel_frames"] = int(inputs["feature_attention_mask"].sum().item())
    if shapes:
        out["shapes"] = shapes

    return out


def format_breakdown(bd):
    """One-line summary for logs, e.g. '8421 tok (vid 6912 / aud 1180 / txt 329)'."""
    def num(x):
        return "?" if x is None else str(x)
    return (f"{bd['total']} tok (vid {num(bd.get('video'))} / "
            f"aud {num(bd.get('audio'))} / txt {num(bd.get('text_and_control'))})")


def generate(model, processor, conversation, use_audio_in_video,
             max_new_tokens=512, temperature=0.0, seed=0):
    import torch

    inputs = prepare_inputs(
        processor, conversation, use_audio_in_video, model.device, model.dtype
    )
    breakdown = token_breakdown(processor, inputs, model)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        return_audio=False,
        use_audio_in_video=use_audio_in_video,
    )
    if temperature and temperature > 0:
        torch.manual_seed(seed)
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)

    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)

    ids = out[0] if isinstance(out, (tuple, list)) else out
    new_tokens = ids[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    breakdown["output"] = int(new_tokens.shape[-1])
    return decoded, breakdown


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_score_json(raw: str):
    """Strict-ish parse. Returns (dict, error). Never guesses a score from prose:
    an unparseable response is recorded as a failure, not rescued by regex."""
    candidate = raw.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start:end + 1]

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(obj, dict):
        return None, "not_an_object"
    if "score" not in obj:
        return None, "missing_score_key"

    score = obj["score"]
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None, f"score_not_numeric: {score!r}"
        if float(score) != int(score):
            return None, f"score_not_integer: {score!r}"
        obj["score"] = int(score)
    return obj, None
