from .ltx_keyframer import LTXKeyframer
from .multi_image_loader import MultiImageLoader
from .ltx_sequencer import LTXSequencer
from .speech_length_calculator import SpeechLengthCalculator
from .load_audio_ui import LoadAudioUI
from .load_video_ui import LoadVideoUI
from .ltx_director import LTXDirector
from .ltx_director_guide import LTXDirectorGuide
from .prompt_relay_lora import LTXPromptRelayEncodeWithTemporalLora
from .frame_spill_nodes import SpillImageBatchToDirectory
from .frame_passthrough_nodes import FrameCountPassthrough
from .frame_transition_nodes import AlignedAudioCutTransition, AlignedOverlapCutTransition, LTXExtensionFrameLogic
from .patches import apply_audio_encode_safety_patch, apply_audio_trim_safety_patch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


apply_audio_encode_safety_patch()
apply_audio_trim_safety_patch()

class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXDirector,
            LTXDirectorGuide,
            LTXPromptRelayEncodeWithTemporalLora,
        ]

async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()
    
NODE_CLASS_MAPPINGS = {
    "LTXKeyframer": LTXKeyframer,
    "MultiImageLoader": MultiImageLoader,
    "LTXSequencer": LTXSequencer,
    "SpeechLengthCalculator": SpeechLengthCalculator,
    "LoadAudioUI": LoadAudioUI,
    "LoadVideoUI": LoadVideoUI,
    "LTXDirectorExtender": LTXDirector,
    "LTXDirectorGuideExtender": LTXDirectorGuide,
    "LTXPromptRelayEncodeWithTemporalLora": LTXPromptRelayEncodeWithTemporalLora,
    "SpillImageBatchToDirectory": SpillImageBatchToDirectory,
    "FrameCountPassthrough": FrameCountPassthrough,
    "AlignedAudioCutTransition": AlignedAudioCutTransition,
    "AlignedOverlapCutTransition": AlignedOverlapCutTransition,
    "LTXExtensionFrameLogic": LTXExtensionFrameLogic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXKeyframer": "LTX Keyframer",
    "MultiImageLoader": "Multi Image Loader",
    "LTXSequencer": "LTX Sequencer",
    "SpeechLengthCalculator": "Speech Length Calculator",
    "LoadAudioUI": "Load Audio UI",
    "LoadVideoUI": "Load Video UI",
    "LTXDirectorExtender": "LTX Director Extender",
    "LTXDirectorGuideExtender": "LTX Director Guide Extender",
    "LTXPromptRelayEncodeWithTemporalLora": "LTX Prompt Relay Encode + Temporal LoRA",
    "SpillImageBatchToDirectory": "Spill Image Batch To Directory",
    "FrameCountPassthrough": "Frame Count Passthrough",
    "AlignedAudioCutTransition": "Aligned Audio Cut Transition",
    "AlignedOverlapCutTransition": "Aligned Overlap Cut Transition",
    "LTXExtensionFrameLogic": "LTX Extension Frame Logic",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']