"""
Legacy compatibility aliases.

The active training/inference path now uses `decoder.py`, which preserves the
official OmniLottie-style architecture and only shifts the Lottie token ID
ranges to sit behind the Qwen3.5 base vocabulary.
"""

from decoder import LottieDecoder as LottieQwen35ForConditionalGeneration

