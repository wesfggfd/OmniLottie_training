from __future__ import annotations

from dataclasses import dataclass

from lottie.objects.lottie_tokenize import LottieTensor


ORIGINAL_BASE_VOCAB_SIZE = 151643
ORIGINAL_VOCAB_SIZE = 192400
ORIGINAL_COMMAND_OFFSET = 151936
ORIGINAL_NUMBER_OFFSET = 173186


@dataclass(frozen=True)
class LottieVocabLayout:
    base_vocab_size: int

    @property
    def lottie_vocab_size(self) -> int:
        return ORIGINAL_VOCAB_SIZE - ORIGINAL_BASE_VOCAB_SIZE

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + self.lottie_vocab_size

    @property
    def shift(self) -> int:
        return self.base_vocab_size - ORIGINAL_BASE_VOCAB_SIZE

    @property
    def command_offset(self) -> int:
        return ORIGINAL_COMMAND_OFFSET + self.shift

    @property
    def number_offset(self) -> int:
        return ORIGINAL_NUMBER_OFFSET + self.shift

    @property
    def bos_token_id(self) -> int:
        return 192398 + self.shift

    @property
    def eos_token_id(self) -> int:
        return 192399 + self.shift

    @property
    def lottie_token_start(self) -> int:
        return ORIGINAL_BASE_VOCAB_SIZE + self.shift

    @property
    def lottie_token_end(self) -> int:
        return ORIGINAL_VOCAB_SIZE - 1 + self.shift

    @property
    def num_commands(self) -> int:
        return len(LottieTensor.COMMANDS)

    def is_lottie_token(self, token_id: int) -> bool:
        return self.lottie_token_start <= token_id <= self.lottie_token_end

    def is_command_token(self, token_id: int) -> bool:
        return self.command_offset <= token_id < self.command_offset + self.num_commands

    def command_token_id(self, command_idx: int) -> int:
        return self.command_offset + command_idx

    def count_token_id(self, count: int) -> int:
        return self.number_offset + count

    def remap_original_lottie_token(self, token_id: int) -> int:
        if ORIGINAL_BASE_VOCAB_SIZE <= token_id < ORIGINAL_VOCAB_SIZE:
            return token_id + self.shift
        return token_id

    def restore_opensource_token(self, token_id: int) -> int:
        if self.is_lottie_token(token_id):
            return token_id - self.shift
        return token_id

    def remap_param_offset(self, command_idx: int, param_idx: int) -> int:
        original_offset = LottieTensor.get_param_offset(command_idx, param_idx)
        if original_offset == 0:
            return 0
        return original_offset + self.shift

