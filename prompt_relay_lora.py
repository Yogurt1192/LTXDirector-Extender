import logging
import types

from typing import Dict, List

import folder_paths
import torch

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)
from .patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)


def _lora_options():
    return [""] + folder_paths.get_filename_list("loras")


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths."""
    if not pixel_lengths:
        return []

    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(value) for value in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda idx: -(exact[idx] - int(exact[idx])))
        for offset in range(diff):
            result[order[offset % len(order)]] += 1

    for idx in range(len(result)):
        if result[idx] >= 1:
            continue
        max_idx = max(range(len(result)), key=lambda pos: result[pos])
        if result[max_idx] > 1:
            result[max_idx] -= 1
            result[idx] = 1

    return result


def parse_float_list(value: str) -> List[float]:
    if not value or not value.strip():
        return []
    try:
        return [float(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    except ValueError as exc:
        raise ValueError(f"Failed to parse float list: {exc}") from exc


def validate_segment_weights(segment_frames: List[int], segment_weights: List[float]) -> None:
    if len(segment_frames) != len(segment_weights):
        raise ValueError(
            f"segment_frames ({len(segment_frames)} values) and lora_segment_weights "
            f"({len(segment_weights)} values) must have the same length"
        )

    for idx, weight in enumerate(segment_weights):
        if not isinstance(weight, (int, float)):
            raise TypeError(f"Segment weight {idx} must be numeric, got {type(weight)}")
        if weight < 0.0 or weight > 1.5:
            log.warning(
                "Segment weight %d is %.3f, recommended range is [0.0, 1.5]. Clamping.",
                idx,
                weight,
            )


def build_temporal_lora_segments(segment_frames: List[int], segment_weights: List[float], fade_frames: int = 0) -> List[Dict]:
    validate_segment_weights(segment_frames, segment_weights)

    segments = []
    cursor = 0
    for idx, (frame_count, weight) in enumerate(zip(segment_frames, segment_weights)):
        frame_count = int(frame_count)
        if frame_count < 0:
            raise ValueError(f"Segment {idx} frame count must be >= 0, got {frame_count}")

        start = cursor
        end = cursor + frame_count
        segments.append({
            "index": idx,
            "start": start,
            "end": end,
            "weight": max(0.0, min(1.5, float(weight))),
            "length": frame_count,
            "fade_frames": fade_frames,
        })
        cursor = end

    return segments


def get_segment_weight(frame_idx: int, segments: List[Dict], fade_frames: int = 0) -> float:
    if not segments:
        return 0.0

    current_seg = None
    next_seg = None
    for idx, segment in enumerate(segments):
        if segment["start"] <= frame_idx < segment["end"]:
            current_seg = segment
            if idx + 1 < len(segments):
                next_seg = segments[idx + 1]
            break

    if current_seg is None:
        return segments[-1]["weight"]

    if fade_frames <= 0 or next_seg is None:
        return current_seg["weight"]

    frames_until_transition = current_seg["end"] - frame_idx
    if frames_until_transition > fade_frames:
        return current_seg["weight"]

    fade_progress = (fade_frames - frames_until_transition) / fade_frames
    fade_progress = max(0.0, min(1.0, fade_progress))
    return current_seg["weight"] + (next_seg["weight"] - current_seg["weight"]) * fade_progress


def build_temporal_lora_mask(latent_frames: int, segments: List[Dict], tokens_per_frame: int, fade_frames: int = 0, debug: bool = False) -> torch.Tensor:
    total_tokens = latent_frames * tokens_per_frame
    mask = torch.zeros(total_tokens, dtype=torch.float32)

    for token_idx in range(total_tokens):
        frame_idx = min(token_idx // tokens_per_frame, latent_frames - 1)
        mask[token_idx] = get_segment_weight(frame_idx, segments, fade_frames)

    mask = mask.view(1, -1, 1)
    if debug:
        log.info(
            "[TemporalLoRA] Built mask: shape %s, weight range [%.4f, %.4f]",
            tuple(mask.shape),
            mask.min().item(),
            mask.max().item(),
        )
    return mask


def load_lora_file(lora_name: str) -> Dict:
    if not lora_name or not lora_name.strip():
        raise ValueError("LoRA name is empty")

    try:
        lora_path = folder_paths.get_full_path("loras", lora_name)
    except Exception as exc:
        raise ValueError(f"LoRA file not found: {lora_name} ({exc})") from exc

    if not lora_path:
        raise ValueError(f"LoRA file not found: {lora_name}")

    from comfy.utils import load_torch_file

    log.info("[TemporalLoRA] Loading LoRA: %s", lora_path)
    return load_torch_file(lora_path)


def validate_temporal_lora_config(segment_frames: List[int], segment_weights: List[float], lora_name: str, total_frames=None, debug: bool = False) -> Dict:
    result = {"valid": True, "warnings": [], "errors": []}

    if len(segment_frames) != len(segment_weights):
        result["valid"] = False
        result["errors"].append(
            f"Mismatch: {len(segment_frames)} segment frames vs {len(segment_weights)} segment weights"
        )

    for idx, weight in enumerate(segment_weights):
        if weight < 0.0 or weight > 1.5:
            result["warnings"].append(f"Segment {idx} weight {weight:.3f} outside [0.0, 1.5]")

    total_configured = sum(segment_frames)
    if total_frames is not None and total_configured != total_frames:
        result["warnings"].append(
            f"Configured frames {total_configured} != latent frames {total_frames}"
        )

    try:
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            result["valid"] = False
            result["errors"].append(f"LoRA file not found: {lora_name}")
    except Exception as exc:
        result["valid"] = False
        result["errors"].append(f"Failed to locate LoRA: {exc}")

    if debug:
        log.info("[TemporalLoRA] Validation: %s", result)
    return result


def _apply_temporal_lora_to_model(model, lora_dict, temporal_mask, latent_frames, model_strength, debug=False):
    import comfy.lora
    import comfy.lora_convert
    import comfy.weight_adapter
    from comfy.weight_adapter.bypass import BypassForwardHook

    converted_lora = comfy.lora_convert.convert_lora(lora_dict)

    key_map = {}
    if model is not None and hasattr(model, "model"):
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    loaded_lora = comfy.lora.load_lora(converted_lora, key_map, log_missing=False)

    if len(loaded_lora) == 0:
        raise ValueError(
            "TemporalLoRA: Selected LoRA matched 0 model patches. "
            "This LoRA is likely incompatible with the current model architecture."
        )

    adapter_patches = {}
    regular_patches = {}
    for key, patch_data in loaded_lora.items():
        if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
            adapter_patches[key] = patch_data
        else:
            regular_patches[key] = patch_data

    patched_model = model
    if regular_patches:
        patched_model.add_patches(regular_patches, model_strength)

    total_tokens = int(temporal_mask.shape[1])
    if latent_frames <= 0:
        raise ValueError("TemporalLoRA: latent_frames must be > 0")

    if total_tokens % latent_frames == 0:
        frame_weights = temporal_mask.view(latent_frames, total_tokens // latent_frames, 1).mean(dim=1).view(-1)
    else:
        idx = torch.floor(torch.arange(latent_frames, dtype=torch.float32) * total_tokens / latent_frames).long()
        idx = torch.clamp(idx, 0, total_tokens - 1)
        frame_weights = temporal_mask.view(-1)[idx]

    token_mask_cache = {}

    def _token_multiplier_for_input(x: torch.Tensor):
        if x is None or x.ndim < 3:
            return model_strength

        tokens = int(x.shape[1])
        key = (tokens, x.device, x.dtype)
        cached = token_mask_cache.get(key)
        if cached is not None:
            return cached

        if tokens == total_tokens:
            token_weights = temporal_mask.view(1, total_tokens, 1)
        elif tokens % latent_frames == 0:
            tokens_per_frame = tokens // latent_frames
            token_weights = frame_weights.repeat_interleave(tokens_per_frame).view(1, tokens, 1)
        else:
            frame_idx = torch.floor(torch.arange(tokens, dtype=torch.float32) * latent_frames / tokens).long()
            frame_idx = torch.clamp(frame_idx, 0, latent_frames - 1)
            token_weights = frame_weights[frame_idx].view(1, tokens, 1)

        scaled = token_weights.to(device=x.device, dtype=x.dtype) * model_strength
        token_mask_cache[key] = scaled
        return scaled

    def _get_module_by_key(root, key):
        module_key = key[:-7] if key.endswith(".weight") else key
        module = root
        for part in module_key.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    hooks = []
    wrapped_modules = 0
    for key, adapter in adapter_patches.items():
        try:
            module = _get_module_by_key(patched_model.model, key)
        except Exception:
            continue

        if not hasattr(module, "weight"):
            continue

        hook = BypassForwardHook(module, adapter, multiplier=1.0)
        hook.inject()
        hooks.append(hook)

        original_bypass_forward = module.forward

        def _temporal_bypass(self_module, x, *args, __orig=original_bypass_forward, __adapter=adapter, **kwargs):
            __adapter.multiplier = _token_multiplier_for_input(x)
            return __orig(x, *args, **kwargs)

        module.forward = types.MethodType(_temporal_bypass, module)
        wrapped_modules += 1

    if not hasattr(patched_model, "_temporal_lora_cache"):
        patched_model._temporal_lora_cache = {"entries": []}

    cache_entry = {
        "temporal_mask": temporal_mask,
        "loaded_lora": loaded_lora,
        "adapter_patch_count": len(adapter_patches),
        "regular_patch_count": len(regular_patches),
        "wrapped_modules": wrapped_modules,
        "hooks": hooks,
        "model_strength": model_strength,
    }
    patched_model._temporal_lora_cache.setdefault("entries", []).append(cache_entry)
    patched_model._temporal_lora_cache["last"] = cache_entry

    if debug:
        log.info(
            "[TemporalLoRA] Loaded %d total patches (%d adapter, %d regular), wrapped %d modules",
            len(loaded_lora),
            len(adapter_patches),
            len(regular_patches),
            wrapped_modules,
        )

    if len(adapter_patches) == 0:
        log.warning(
            "[TemporalLoRA] LoRA resolved without bypass adapters; temporal token-wise scaling may be limited."
        )

    return patched_model


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    for name, val in (("global_prompt", global_prompt), ("local_prompts", local_prompts), ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    locals_list = [prompt.strip() for prompt in local_prompts.split("|")]
    for prompt in locals_list:
        if not prompt:
            raise ValueError("There is a segment on the timeline missing a prompt!")

    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        raise ValueError("At least one local prompt is required.")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)
    return patched, conditioning, effective_lengths, latent_frames, tokens_per_frame


class LTXPromptRelayEncodeWithTemporalLora(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXPromptRelayEncodeWithTemporalLora",
            display_name="LTX Prompt Relay Encode + Temporal LoRA",
            category="conditioning/prompt_relay",
            description=(
                "Standalone Prompt Relay encode node with per-segment Temporal LoRA controls. "
                "This is kept separate from the tested LTXDirector extension path."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input("global_prompt", multiline=True, default=""),
                io.String.Input("local_prompts", multiline=True, default=""),
                io.String.Input("segment_lengths", default=""),
                io.Float.Input("epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4),
                io.Combo.Input("lora_name", options=_lora_options(), tooltip="LoRA slot 1. Leave blank to disable."),
                io.String.Input("lora_segment_weights", default=""),
                io.Float.Input("lora_model_strength", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("lora_clip_strength", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Int.Input("lora_fade_frames", default=0, min=0, max=200, step=1),
                io.Combo.Input("lora_name_2", options=_lora_options(), tooltip="LoRA slot 2. Leave blank to disable."),
                io.String.Input("lora_segment_weights_2", default=""),
                io.Float.Input("lora_model_strength_2", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("lora_clip_strength_2", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Int.Input("lora_fade_frames_2", default=0, min=0, max=200, step=1),
                io.Combo.Input("lora_name_3", options=_lora_options(), tooltip="LoRA slot 3. Leave blank to disable."),
                io.String.Input("lora_segment_weights_3", default=""),
                io.Float.Input("lora_model_strength_3", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("lora_clip_strength_3", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Int.Input("lora_fade_frames_3", default=0, min=0, max=200, step=1),
                io.Combo.Input("lora_name_4", options=_lora_options(), tooltip="LoRA slot 4. Leave blank to disable."),
                io.String.Input("lora_segment_weights_4", default=""),
                io.Float.Input("lora_model_strength_4", default=1.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("lora_clip_strength_4", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Int.Input("lora_fade_frames_4", default=0, min=0, max=200, step=1),
                io.Boolean.Input("debug_temporal_lora", default=False),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        latent,
        global_prompt,
        local_prompts,
        segment_lengths,
        epsilon,
        lora_name,
        lora_segment_weights,
        lora_model_strength,
        lora_clip_strength,
        lora_fade_frames,
        lora_name_2,
        lora_segment_weights_2,
        lora_model_strength_2,
        lora_clip_strength_2,
        lora_fade_frames_2,
        lora_name_3,
        lora_segment_weights_3,
        lora_model_strength_3,
        lora_clip_strength_3,
        lora_fade_frames_3,
        lora_name_4,
        lora_segment_weights_4,
        lora_model_strength_4,
        lora_clip_strength_4,
        lora_fade_frames_4,
        debug_temporal_lora,
    ) -> io.NodeOutput:
        patched_model, conditioning, segment_frames, latent_frames, tokens_per_frame = _encode_relay(
            model,
            clip,
            latent,
            global_prompt,
            local_prompts,
            segment_lengths,
            epsilon,
        )

        lora_configs = [
            (1, lora_name, lora_segment_weights, lora_model_strength, lora_clip_strength, lora_fade_frames),
            (2, lora_name_2, lora_segment_weights_2, lora_model_strength_2, lora_clip_strength_2, lora_fade_frames_2),
            (3, lora_name_3, lora_segment_weights_3, lora_model_strength_3, lora_clip_strength_3, lora_fade_frames_3),
            (4, lora_name_4, lora_segment_weights_4, lora_model_strength_4, lora_clip_strength_4, lora_fade_frames_4),
        ]

        applied_count = 0
        for slot, slot_lora_name, slot_weights_text, slot_model_strength, slot_clip_strength, slot_fade in lora_configs:
            if not slot_lora_name:
                continue

            if slot_clip_strength != 0.0 and debug_temporal_lora:
                log.warning(
                    "[PromptRelayWithTemporalLora] LoRA slot %d has non-zero clip strength %.3f, but clip temporal modulation is not implemented.",
                    slot,
                    slot_clip_strength,
                )

            slot_weights = parse_float_list(slot_weights_text)
            if not slot_weights:
                if debug_temporal_lora:
                    log.info("[PromptRelayWithTemporalLora] LoRA slot %d has no segment weights, skipping.", slot)
                continue

            validation = validate_temporal_lora_config(
                segment_frames,
                slot_weights,
                slot_lora_name,
                sum(segment_frames),
                debug_temporal_lora,
            )
            if not validation["valid"]:
                raise ValueError(f"Temporal LoRA slot {slot} config invalid: {validation['errors']}")

            # Build per-frame weights from the same segment schedule used by PromptRelay.
            segments = build_temporal_lora_segments(segment_frames, slot_weights, slot_fade)
            temporal_mask = build_temporal_lora_mask(
                latent_frames,
                segments,
                tokens_per_frame,
                slot_fade,
                debug_temporal_lora,
            )

            lora_dict = load_lora_file(slot_lora_name)
            patched_model = _apply_temporal_lora_to_model(
                patched_model,
                lora_dict,
                temporal_mask,
                latent_frames,
                slot_model_strength,
                debug_temporal_lora,
            )
            applied_count += 1

            if debug_temporal_lora:
                log.info(
                    "[PromptRelayWithTemporalLora] Applied temporal LoRA slot %d: %s with %d segments, fade_frames=%d",
                    slot,
                    slot_lora_name,
                    len(segments),
                    slot_fade,
                )

        if debug_temporal_lora and applied_count == 0:
            log.info("[PromptRelayWithTemporalLora] No temporal LoRA slots applied.")

        return io.NodeOutput(patched_model, conditioning)