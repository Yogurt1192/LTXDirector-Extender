import math
import types
import torch
import comfy.ldm.modules.attention


_AUDIO_ENCODE_PATCHED = False
_AUDIO_TRIM_PATCHED = False


def _audio_vae_min_samples(vae):
    first_stage_model = getattr(vae, "first_stage_model", None)
    preprocessor = getattr(first_stage_model, "preprocessor", None)
    n_fft = getattr(preprocessor, "n_fft", None)
    if n_fft is not None:
        return max(1, int(n_fft))
    return 1024


def _pad_audio_tail(waveform, minimum_samples):
    sample_count = waveform.shape[-1]
    if sample_count >= minimum_samples:
        return waveform

    pad = waveform.new_zeros(*waveform.shape[:-1], minimum_samples - sample_count)
    return torch.cat((waveform, pad), dim=-1)


def _silent_audio_like(waveform, sample_rate, sample_count=1):
    channels = waveform.shape[1] if waveform.ndim >= 2 else 2
    batch = waveform.shape[0] if waveform.ndim >= 1 else 1
    silent = waveform.new_zeros((batch, channels, max(1, int(sample_count))))
    return {"waveform": silent, "sample_rate": sample_rate}


def _ltx_audio_target_sample_rate(vae):
    first_stage_model = getattr(vae, "first_stage_model", None)
    preprocessor = getattr(first_stage_model, "preprocessor", None)
    target_sample_rate = getattr(preprocessor, "target_sample_rate", None)
    if target_sample_rate is not None:
        return int(target_sample_rate)
    sample_rate = getattr(first_stage_model, "sample_rate", None)
    if sample_rate is not None:
        return int(sample_rate)
    return None


def apply_audio_encode_safety_patch():
    global _AUDIO_ENCODE_PATCHED
    if _AUDIO_ENCODE_PATCHED:
        return

    import torchaudio
    from comfy_api.latest import IO
    from comfy_extras.nodes_audio import VAEEncodeAudio

    @classmethod
    def _safe_execute(cls, vae, audio):
        if audio is None:
            raise ValueError("VAEEncodeAudio: input audio is None (source video may have no audio track).")

        sample_rate = audio["sample_rate"]
        first_stage_model = getattr(vae, "first_stage_model", None)
        ltx_target_sample_rate = _ltx_audio_target_sample_rate(vae)
        vae_sample_rate = ltx_target_sample_rate or getattr(vae, "audio_sample_rate", 44100)
        minimum_target_samples = _audio_vae_min_samples(vae)
        minimum_source_samples = max(1, math.ceil(minimum_target_samples * sample_rate / max(vae_sample_rate, 1)))
        waveform = audio["waveform"].contiguous()
        waveform = _pad_audio_tail(waveform, minimum_source_samples)

        if ltx_target_sample_rate is None and vae_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)

        waveform = _pad_audio_tail(waveform.contiguous(), minimum_target_samples)

        if first_stage_model is not None and ltx_target_sample_rate is not None:
            encoded = first_stage_model.encode(waveform, sample_rate=sample_rate)
        else:
            encoded = vae.encode(waveform.movedim(1, -1).contiguous())
        return IO.NodeOutput({"samples": encoded})

    VAEEncodeAudio.execute = _safe_execute
    VAEEncodeAudio.encode = _safe_execute
    _AUDIO_ENCODE_PATCHED = True


def apply_audio_trim_safety_patch():
    global _AUDIO_TRIM_PATCHED
    if _AUDIO_TRIM_PATCHED:
        return

    from comfy_api.latest import IO
    from comfy_extras.nodes_audio import TrimAudioDuration

    @classmethod
    def _safe_trim_execute(cls, audio, start_index, duration):
        if audio is None:
            return IO.NodeOutput(None)

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        audio_length = waveform.shape[-1]

        if audio_length == 0:
            return IO.NodeOutput(_silent_audio_like(waveform, sample_rate))

        if start_index < 0:
            start_frame = audio_length + int(round(start_index * sample_rate))
        else:
            start_frame = int(round(start_index * sample_rate))
        start_frame = max(0, min(start_frame, audio_length))

        duration_frames = int(round(duration * sample_rate))
        end_frame = max(0, min(start_frame + duration_frames, audio_length))

        if start_frame >= end_frame:
            return IO.NodeOutput(_silent_audio_like(waveform, sample_rate))

        trimmed = waveform[..., start_frame:end_frame].contiguous()
        return IO.NodeOutput({"waveform": trimmed, "sample_rate": sample_rate})

    TrimAudioDuration.execute = _safe_trim_execute
    TrimAudioDuration.trim = _safe_trim_execute
    _AUDIO_TRIM_PATCHED = True


def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    # Bypass wrap_attn (sage/etc may ignore masks) by calling attention_pytorch directly.
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )


def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )
    return self.o(x)


def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]

    q = self.norm_q(self.q(x))

    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(
        q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options,
    )

    k = self.norm_k(self.k(context_text))
    v = self.v(context_text)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )

    return self.o(x + img_x)


def _ltx_forward(self, mask_fn, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    from comfy.ldm.lightricks.model import apply_rotary_emb

    is_self_attn = context is None
    context = x if is_self_attn else context

    q = self.q_norm(self.to_q(x))
    k = self.k_norm(self.to_k(context))
    v = self.to_v(context)

    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    if not is_self_attn:
        temporal_mask = mask_fn(q, k, transformer_options)
        if temporal_mask is not None:
            mask = temporal_mask if mask is None else mask + temporal_mask

    if mask is None:
        out = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, self.heads, attn_precision=self.attn_precision,
            transformer_options=transformer_options,
        )
    else:
        out = _masked_attention(q, k, v, self.heads, mask=mask,
                                attn_precision=self.attn_precision,
                                transformer_options=transformer_options)

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)


class _CrossAttnPatch:
    """Descriptor that binds (impl, mask_fn) as a method onto a cross-attn module."""

    def __init__(self, impl, mask_fn):
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl, mask_fn = self.impl, self.mask_fn

        def wrapped(self_module, *args, **kwargs):
            return impl(self_module, mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


def detect_model_type(model):
    """Return (arch, patch_size, temporal_stride) for latent geometry.

    temporal_stride is the VAE's pixel→latent temporal compression factor,
    used to convert user-facing pixel frame counts to latent frames.
    """
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        return "wan", tuple(diff_model.patch_size), 4

    if hasattr(diff_model, "patchifier"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"Unsupported model type: {type(diff_model).__name__}. "
        f"Currently supports Wan and LTX models."
    )


def _check_unpatched(model_clone, key):
    if key in getattr(model_clone, "object_patches", {}):
        raise RuntimeError(
            f"PromptRelay: cross-attention forward at '{key}' is already patched by "
            "another node (e.g. KJNodes NAG). Stacking is not supported — remove the "
            "conflicting node."
        )


def apply_patches(model_clone, arch, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            key = f"diffusion_model.blocks.{idx}.cross_attn.forward"
            _check_unpatched(model_clone, key)
            cross_attn = block.cross_attn
            impl = _wan_i2v_forward if isinstance(cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(key, _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__))
        return

    if arch == "ltx":
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                _check_unpatched(model_clone, key)
                model_clone.add_object_patch(key, _CrossAttnPatch(_ltx_forward, mask_fn).__get__(module, module.__class__))
        return

    raise ValueError(f"Unknown model arch: {arch}")
