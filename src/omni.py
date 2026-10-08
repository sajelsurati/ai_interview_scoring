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


def prepare_inputs(processor, conversation, use_audio_in_video, device, dtype):
    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text, audio=audios, image=images, video=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(device)
    if dtype is not None:
        # BatchFeature.to(dtype) casts only floating-point tensors, so token ids
        # stay integral.
        inputs = inputs.to(dtype)
    return inputs


def generate(model, processor, conversation, use_audio_in_video,
             max_new_tokens=512, temperature=0.0, seed=0):
    import torch

    inputs = prepare_inputs(
        processor, conversation, use_audio_in_video, model.device, model.dtype
    )
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
    return decoded, int(inputs["input_ids"].shape[1])


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
