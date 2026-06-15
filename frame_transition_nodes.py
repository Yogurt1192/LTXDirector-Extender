import torch


class AlignedOverlapCutTransition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE",),
                "new_images": ("IMAGE",),
                "overlap": ("INT", {"default": 12, "min": 1, "max": 4096, "step": 1}),
                "edge_margin": ("INT", {"default": 1, "min": 0, "max": 256, "step": 1}),
                "blend_frames": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "seam_shift": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "seam_index", "anchor_frame", "anchor_frames_tail")
    FUNCTION = "transition"
    CATEGORY = "LTXDirector/transition"

    @classmethod
    def IS_CHANGED(cls, source_images, new_images, overlap, edge_margin, blend_frames, seam_shift):
        # Force the seam splice to recompute while transition tuning is active,
        # so subgraph or prompt caching cannot mask blend and seam-shift changes.
        return float("nan")

    @staticmethod
    def _frame_mse(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (first - second).pow(2).mean(dim=(1, 2, 3))

    def _infer_overlap(self, source_images: torch.Tensor, new_images: torch.Tensor, requested_overlap: int) -> int:
        source_count = int(source_images.shape[0])
        new_count = int(new_images.shape[0])
        max_overlap = min(source_count, new_count)
        if max_overlap <= 1:
            return max_overlap

        if requested_overlap > 1:
            return min(requested_overlap, max_overlap)

        candidate_cap = min(max_overlap, 64)
        best_overlap = 1
        best_score = None
        for candidate in range(2, candidate_cap + 1):
            source_window = source_images[-candidate:]
            new_window = new_images[:candidate]
            score = float(self._frame_mse(source_window, new_window).mean())
            if best_score is None or score < best_score:
                best_score = score
                best_overlap = candidate

        return best_overlap

    def _score_seam(self, source_overlap: torch.Tensor, new_overlap: torch.Tensor, seam_index: int, overlap: int) -> float:
        boundary_score = float(self._frame_mse(source_overlap[seam_index - 1 : seam_index], new_overlap[seam_index : seam_index + 1])[0])

        window = min(3, seam_index, overlap - seam_index)
        if window > 0:
            left_window = source_overlap[seam_index - window : seam_index]
            right_window = new_overlap[seam_index : seam_index + window]
            window_score = float(self._frame_mse(left_window, right_window).mean())
        else:
            window_score = boundary_score

        # Favor later seams slightly so dynamic regions stay anchored to the real
        # source overlap instead of cutting into regenerated overlap too early.
        seam_penalty = (overlap - seam_index) / max(overlap, 1)
        return boundary_score + (0.5 * window_score) + (0.03 * seam_penalty)

    def transition(self, source_images, new_images, overlap, edge_margin, blend_frames, seam_shift):
        if source_images is None or new_images is None:
            raise ValueError("source_images and new_images are required")
        if source_images.shape[1:3] != new_images.shape[1:3]:
            raise ValueError(
                f"Source and new images must have the same shape: {source_images.shape[1:3]} vs {new_images.shape[1:3]}"
            )

        requested_overlap = int(overlap)
        overlap = self._infer_overlap(source_images, new_images, requested_overlap)
        if overlap <= 0:
            merged = torch.cat((source_images, new_images), dim=0)
            anchor_frame = merged[-1:]
            return (merged, 0, anchor_frame, anchor_frame)

        source_overlap = source_images[-overlap:]
        new_overlap = new_images[:overlap]

        if overlap == 1:
            merged = torch.cat((source_images, new_images[1:]), dim=0)
            anchor_frame = merged[-1:]
            return (merged, 1, anchor_frame, anchor_frame)

        edge_margin = max(0, min(int(edge_margin), (overlap - 1) // 2))
        candidate_start = max(1, edge_margin)
        candidate_end = min(overlap - 1, overlap - edge_margin)

        if candidate_start > candidate_end:
            seam_index = overlap // 2
        else:
            seam_index = min(
                range(candidate_start, candidate_end + 1),
                key=lambda candidate: self._score_seam(source_overlap, new_overlap, candidate, overlap),
            )

        seam_shift = int(seam_shift)
        seam_index = max(candidate_start, min(candidate_end, seam_index + seam_shift))

        source_keep = source_images.shape[0] - overlap + seam_index
        merged = torch.cat((source_images[:source_keep], new_images[seam_index:]), dim=0)

        blend_frames = max(0, int(blend_frames))
        pre_blend_window = min(blend_frames, source_keep, new_images.shape[0] - seam_index)
        if pre_blend_window > 0:
            blend_start = source_keep - pre_blend_window
            left = source_images[blend_start:source_keep]
            right = new_images[seam_index : seam_index + pre_blend_window]
            if left.shape[0] == pre_blend_window and right.shape[0] == pre_blend_window:
                # Blend into the seam over the last kept source frames so the
                # approach to the cut starts matching the new continuation.
                weights = torch.linspace(
                    1.0 / (pre_blend_window + 1),
                    pre_blend_window / (pre_blend_window + 1),
                    pre_blend_window,
                    device=merged.device,
                    dtype=merged.dtype,
                )
                weights = weights.view(-1, 1, 1, 1)
                merged[blend_start:source_keep] = (left * (1.0 - weights)) + (right * weights)

        post_blend_window = min(blend_frames, source_images.shape[0] - source_keep, merged.shape[0] - source_keep)
        if post_blend_window > 0:
            left = source_images[source_keep : source_keep + post_blend_window]
            right = merged[source_keep : source_keep + post_blend_window]
            if left.shape[0] == post_blend_window and right.shape[0] == post_blend_window:
                # Also soften the first kept new frames, since that is often where
                # the discontinuity is most visible.
                weights = torch.linspace(
                    1.0 / (post_blend_window + 1),
                    post_blend_window / (post_blend_window + 1),
                    post_blend_window,
                    device=merged.device,
                    dtype=merged.dtype,
                )
                weights = weights.view(-1, 1, 1, 1)
                merged[source_keep : source_keep + post_blend_window] = (left * (1.0 - weights)) + (right * weights)

        anchor_frame = merged[source_keep - 1 : source_keep]
        anchor_tail_length = max(1, overlap - seam_index + 1)
        anchor_frames_tail = merged[source_keep - 1 : source_keep - 1 + anchor_tail_length]
        return (merged, seam_index, anchor_frame, anchor_frames_tail)


class AlignedAudioCutTransition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",),
                "new_audio": ("AUDIO",),
                "overlap_frames": ("INT", {"default": 12, "min": 0, "max": 4096, "step": 1}),
                "seam_index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "seam_seconds")
    FUNCTION = "transition"
    CATEGORY = "LTXDirector/transition"

    @staticmethod
    def _match_channels(source_waveform: torch.Tensor, new_waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_channels = source_waveform.shape[1]
        new_channels = new_waveform.shape[1]

        if source_channels == new_channels:
            return source_waveform, new_waveform

        if source_channels == 1 and new_channels == 2:
            return source_waveform.repeat(1, 2, 1), new_waveform
        if source_channels == 2 and new_channels == 1:
            return source_waveform, new_waveform.repeat(1, 2, 1)

        raise ValueError(f"Unsupported channel mismatch: {source_channels} vs {new_channels}")

    def transition(self, source_audio, new_audio, overlap_frames, seam_index, frame_rate):
        if source_audio is None:
            return (new_audio, 0.0)
        if new_audio is None:
            return (source_audio, 0.0)
        if frame_rate <= 0:
            raise ValueError("frame_rate must be greater than zero")

        from comfy_extras.nodes_audio import match_audio_sample_rates

        source_waveform = source_audio["waveform"]
        new_waveform = new_audio["waveform"]
        source_sample_rate = source_audio["sample_rate"]
        new_sample_rate = new_audio["sample_rate"]

        source_waveform, new_waveform = self._match_channels(source_waveform, new_waveform)
        source_waveform, new_waveform, sample_rate = match_audio_sample_rates(
            source_waveform,
            source_sample_rate,
            new_waveform,
            new_sample_rate,
        )

        overlap_frames = max(0, int(overlap_frames))
        seam_index = max(0, min(int(seam_index), overlap_frames))
        overlap_samples = int(round((overlap_frames / frame_rate) * sample_rate))
        seam_samples = int(round((seam_index / frame_rate) * sample_rate))

        source_keep = max(0, min(source_waveform.shape[-1], source_waveform.shape[-1] - overlap_samples + seam_samples))
        new_start = max(0, min(new_waveform.shape[-1], seam_samples))

        merged = torch.cat((source_waveform[..., :source_keep], new_waveform[..., new_start:]), dim=-1)
        seam_seconds = seam_index / frame_rate
        return ({"waveform": merged, "sample_rate": sample_rate}, float(seam_seconds))


class LTXExtensionFrameLogic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared_images": ("IMAGE",),
                "extend_frames": ("INT", {"default": 125, "min": 0, "max": 65535, "step": 1}),
                "overlap_frames": ("INT", {"default": 25, "min": 0, "max": 65535, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = (
        "overlap_seconds",
        "extend_frames",
        "extend_seconds",
        "overlap_extend_seconds",
        "accumulated_duration_seconds",
        "ref_audio_start_seconds",
        "ref_audio_duration_seconds",
        "ref_frame_at_end",
        "ref_frames_tail",
        "frame_count",
    )
    FUNCTION = "compute"
    CATEGORY = "LTXDirector/transition"

    def compute(self, prepared_images, extend_frames, overlap_frames, frame_rate):
        if prepared_images is None:
            raise ValueError("prepared_images is required")
        if frame_rate <= 0:
            raise ValueError("frame_rate must be greater than zero")

        frame_count = int(prepared_images.shape[0])
        extend_frames = max(0, int(extend_frames))
        overlap_frames = max(0, min(int(overlap_frames), frame_count))

        overlap_seconds = overlap_frames / frame_rate
        extend_seconds = extend_frames / frame_rate
        overlap_extend_seconds = overlap_seconds + extend_seconds
        accumulated_duration_seconds = frame_count / frame_rate
        ref_audio_start_seconds = max(0.0, accumulated_duration_seconds - overlap_seconds)
        ref_audio_duration_seconds = overlap_seconds

        if frame_count == 0:
            raise ValueError("prepared_images must contain at least one frame")

        ref_frame_at_end = prepared_images[-1:]
        ref_frames_tail = prepared_images[-overlap_frames:] if overlap_frames > 0 else prepared_images[-1:]

        return (
            float(overlap_seconds),
            extend_frames,
            float(extend_seconds),
            float(overlap_extend_seconds),
            float(accumulated_duration_seconds),
            float(ref_audio_start_seconds),
            float(ref_audio_duration_seconds),
            ref_frame_at_end,
            ref_frames_tail,
            frame_count,
        )