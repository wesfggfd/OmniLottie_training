from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig, Qwen3_5ForConditionalGeneration

from tokenizer.offset_vocab import LottieVocabLayout


class HybridInputEmbedding(nn.Module):
    def __init__(self, base_embedding: nn.Module, num_new_tokens: int, hidden_size: int):
        super().__init__()
        self.base_embedding = base_embedding
        self.base_embedding.requires_grad_(False)
        self.base_vocab_size = base_embedding.num_embeddings
        self.lottie_embedding = nn.Embedding(num_new_tokens, hidden_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        clipped = input_ids.clamp_max(self.base_vocab_size - 1)
        hidden_states = self.base_embedding(clipped)

        lottie_mask = input_ids >= self.base_vocab_size
        if lottie_mask.any():
            lottie_ids = input_ids[lottie_mask] - self.base_vocab_size
            hidden_states[lottie_mask] = self.lottie_embedding(lottie_ids).to(hidden_states.dtype)
        return hidden_states


class HybridLMHead(nn.Module):
    def __init__(self, base_head: nn.Module, hidden_size: int, num_new_tokens: int):
        super().__init__()
        self.base_head = base_head
        self.base_head.requires_grad_(False)
        self.lottie_head = nn.Linear(hidden_size, num_new_tokens, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_head(hidden_states)
        lottie_logits = self.lottie_head(hidden_states).to(base_logits.dtype)
        return torch.cat([base_logits, lottie_logits], dim=-1)


class LottieQwen35Config(PretrainedConfig):
    model_type = "lottie_qwen35"

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen3.5-9B",
        pix_len: int = 4560,
        text_len: int = 1500,
        vocab_size: int = 289077,
        bos_token_id: int = 289075,
        eos_token_id: int = 289076,
        pad_token_id: int = 248044,
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


class LottieQwen35ForConditionalGeneration(PreTrainedModel):
    config_class = LottieQwen35Config
    base_model_prefix = "lottie_qwen35"
    supports_gradient_checkpointing = True

    def __init__(self, config: LottieQwen35Config):
        super().__init__(config)
        self.config = config
        self.transformer = None
        self.base_vocab_size = None
        self.lottie_vocab_layout = None
        self._build_model()

    def _build_model(self) -> None:
        model_path = self.config.base_model_path
        qwen_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        original_base_vocab_size = qwen_config.text_config.vocab_size
        qwen_config.bos_token_id = self.config.bos_token_id
        qwen_config.eos_token_id = self.config.eos_token_id
        qwen_config.pad_token_id = self.config.pad_token_id

        self.transformer = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config=qwen_config,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
            ignore_mismatched_sizes=True,
        )

        self.base_vocab_size = original_base_vocab_size
        self.lottie_vocab_layout = LottieVocabLayout(base_vocab_size=self.base_vocab_size)
        hidden_size = self.transformer.config.text_config.hidden_size
        num_new_tokens = self.lottie_vocab_layout.lottie_vocab_size

        base_embedding = self.transformer.get_input_embeddings()
        base_lm_head = self.transformer.get_output_embeddings()
        hybrid_embedding = HybridInputEmbedding(base_embedding, num_new_tokens, hidden_size)
        hybrid_lm_head = HybridLMHead(base_lm_head, hidden_size, num_new_tokens)

        init_std = getattr(self.transformer.config.text_config, "initializer_range", 0.02)
        nn.init.normal_(hybrid_embedding.lottie_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(hybrid_lm_head.lottie_head.weight, mean=0.0, std=init_std)

        self.transformer.set_input_embeddings(hybrid_embedding)
        self.transformer.set_output_embeddings(hybrid_lm_head)
        self.transformer.config.vocab_size = self.config.vocab_size
        self.transformer.config.text_config.vocab_size = self.config.vocab_size

    @staticmethod
    def default_lora_target_modules(include_k_proj: bool = False) -> list[str]:
        modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if include_k_proj:
            modules.insert(1, "k_proj")
        return modules

    def lottie_trainable_parameters(self) -> Iterable[nn.Parameter]:
        input_embeddings = self.transformer.get_input_embeddings()
        output_embeddings = self.transformer.get_output_embeddings()
        yield input_embeddings.lottie_embedding.weight
        yield output_embeddings.lottie_head.weight

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.transformer.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def get_output_embeddings(self):
        return self.transformer.get_output_embeddings()

