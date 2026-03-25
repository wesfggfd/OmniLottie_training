"""
HF-compatible config that mirrors the official-style decoder entrypoint.
"""

from transformers import PretrainedConfig


class LottieDecoderConfig(PretrainedConfig):
    model_type = "lottie_decoder"

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen3.5-9B",
        pix_len: int = 4560,
        text_len: int = 1500,
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.base_model_path = base_model_path
        self.pix_len = pix_len
        self.text_len = text_len
        self.vocab_size = vocab_size
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
