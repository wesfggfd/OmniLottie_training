import os

import torch
from transformers import AutoConfig, PreTrainedModel, Qwen3_5ForConditionalGeneration

from configuration_lottie_decoder import LottieDecoderConfig
from tokenizer.offset_vocab import LottieVocabLayout


class LottieDecoder(PreTrainedModel):
    """
    HF-compatible decoder that preserves the official-style architecture:
    resize the base model vocab to append the shifted Lottie token space.
    """

    config_class = LottieDecoderConfig
    base_model_prefix = "lottie_decoder"
    supports_gradient_checkpointing = True

    def __init__(self, config: LottieDecoderConfig):
        super().__init__(config)
        self.config = config

        base_config = AutoConfig.from_pretrained(config.base_model_path, trust_remote_code=True)
        base_vocab_size = getattr(getattr(base_config, "text_config", base_config), "vocab_size")
        layout = LottieVocabLayout(base_vocab_size=base_vocab_size)

        self.config.vocab_size = self.config.vocab_size or layout.vocab_size
        self.config.bos_token_id = self.config.bos_token_id or layout.bos_token_id
        self.config.eos_token_id = self.config.eos_token_id or layout.eos_token_id
        self.config.pad_token_id = self.config.pad_token_id or layout.pad_token_id

        qwen_config = AutoConfig.from_pretrained(
            config.base_model_path,
            vocab_size=self.config.vocab_size,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            trust_remote_code=True,
        )
        if hasattr(qwen_config, "text_config"):
            qwen_config.text_config.vocab_size = self.config.vocab_size

        dtype = getattr(torch, self.config.torch_dtype) if isinstance(self.config.torch_dtype, str) else self.config.torch_dtype
        self.transformer = Qwen3_5ForConditionalGeneration.from_pretrained(
            config.base_model_path,
            config=qwen_config,
            torch_dtype=dtype,
            attn_implementation=self.config.attn_implementation,
            ignore_mismatched_sizes=True,
        )
        self.transformer.resize_token_embeddings(self.config.vocab_size)
        self.transformer.config.vocab_size = self.config.vocab_size
        if hasattr(self.transformer.config, "text_config"):
            self.transformer.config.text_config.vocab_size = self.config.vocab_size

        self.vocab_size = self.config.vocab_size
        self.bos_token_id = self.config.bos_token_id
        self.eos_token_id = self.config.eos_token_id
        self.pad_token_id = self.config.pad_token_id

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load LottieDecoder from pretrained model path

        Supports two loading methods:
        1. Load from Hugging Face standard format (recommended)
        2. Load from old format pytorch_model.bin (backward compatible)
        """
        if os.path.isdir(pretrained_model_name_or_path):
            old_format_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(old_format_path) and not os.path.exists(os.path.join(pretrained_model_name_or_path, "config.json")):
                print(f"Detected old format model, loading from {old_format_path}...")
                return cls._from_old_format(pretrained_model_name_or_path, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def _from_old_format(cls, checkpoint_path, **kwargs):
        pix_len = kwargs.pop("pix_len", 4560)
        text_len = kwargs.pop("text_len", 1500)
        base_model_path = kwargs.pop("base_model_path", "Qwen/Qwen3.5-9B")
        config = LottieDecoderConfig(
            pix_len=pix_len,
            text_len=text_len,
            base_model_path=base_model_path,
        )
        model = cls(config)
        model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded weights from {model_file}")
        else:
            print(f"Warning: Model file not found {model_file}")
        return model

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.transformer.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def get_output_embeddings(self):
        return self.transformer.get_output_embeddings()
