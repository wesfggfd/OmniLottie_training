import os

import torch
import torch.nn as nn
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

from tokenizer.offset_vocab import LottieVocabLayout


class LottieDecoder(nn.Module):
    """
    OmniLottie-style decoder that preserves the original "resize the model
    vocab" architecture while shifting all Lottie token IDs behind the
    Qwen3.5 base vocabulary.
    """

    def __init__(
        self,
        pix_len,
        text_len,
        model_path="Qwen/Qwen3.5-9B",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        **kwargs,
    ):
        super().__init__()

        self.pix_len = pix_len
        self.text_len = text_len
        self.model_path = model_path

        base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        base_vocab_size = getattr(getattr(base_config, "text_config", base_config), "vocab_size")
        self.vocab_layout = LottieVocabLayout(base_vocab_size=base_vocab_size)

        self.base_vocab_size = base_vocab_size
        self.vocab_size = self.vocab_layout.vocab_size
        self.bos_token_id = self.vocab_layout.bos_token_id
        self.eos_token_id = self.vocab_layout.eos_token_id
        self.pad_token_id = self.vocab_layout.pad_token_id

        print(f"Loading model from {model_path}...")

        config = AutoConfig.from_pretrained(
            model_path,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            trust_remote_code=True,
        )
        if hasattr(config, "text_config"):
            config.text_config.vocab_size = self.vocab_size

        self.transformer = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            ignore_mismatched_sizes=True,
        )
        self.transformer.resize_token_embeddings(self.vocab_size)
        self._register_lottie_gradient_masks()
        self.train()

    @staticmethod
    def default_lora_target_modules(include_k_proj: bool = False) -> list[str]:
        modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if include_k_proj:
            modules.insert(1, "k_proj")
        return modules

    def _register_lottie_gradient_masks(self) -> None:
        def lottie_only_rows(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[: self.base_vocab_size].zero_()
            return grad

        self.transformer.get_input_embeddings().weight.register_hook(lottie_only_rows)
        self.transformer.get_output_embeddings().weight.register_hook(lottie_only_rows)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.transformer.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def get_output_embeddings(self):
        return self.transformer.get_output_embeddings()

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))