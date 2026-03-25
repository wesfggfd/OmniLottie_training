from __future__ import annotations

from dataclasses import dataclass

from lottie.objects.lottie_tokenize import LottieTensor


ORIGINAL_BASE_VOCAB_SIZE = 151643
ORIGINAL_VOCAB_SIZE = 192400
ORIGINAL_COMMAND_OFFSET = 151936
ORIGINAL_NUMBER_OFFSET = 173186
ORIGINAL_PAD_TOKEN_ID = 151643
ORIGINAL_BOS_TOKEN_ID = 192398
ORIGINAL_EOS_TOKEN_ID = 192399


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
    def pad_token_id(self) -> int:
        return ORIGINAL_PAD_TOKEN_ID + self.shift

    @property
    def bos_token_id(self) -> int:
        return ORIGINAL_BOS_TOKEN_ID + self.shift

    @property
    def eos_token_id(self) -> int:
        return ORIGINAL_EOS_TOKEN_ID + self.shift

    @property
    def lottie_token_start(self) -> int:
        return ORIGINAL_BASE_VOCAB_SIZE + self.shift

    @property
    def lottie_token_end(self) -> int:
        return ORIGINAL_VOCAB_SIZE - 1 + self.shift

    @property
    def num_commands(self) -> int:
        return len(LottieTensor.COMMANDS)

    @property
    def lottie_special_token_ids(self) -> tuple[int, int]:
        return (self.bos_token_id, self.eos_token_id)

    def is_lottie_token(self, token_id: int) -> bool:
        return self.lottie_token_start <= token_id <= self.lottie_token_end

    def is_command_token(self, token_id: int) -> bool:
        return self.command_offset <= token_id < self.command_offset + self.num_commands

    def is_count_token(self, token_id: int) -> bool:
        return self.number_offset <= token_id < self.number_offset + 512

    def command_token_id(self, command_idx: int) -> int:
        return self.command_offset + command_idx

    def count_token_id(self, count: int) -> int:
        return self.number_offset + count

    def remap_original_special_token(self, token_id: int) -> int:
        if token_id == ORIGINAL_PAD_TOKEN_ID:
            return self.pad_token_id
        if token_id == ORIGINAL_BOS_TOKEN_ID:
            return self.bos_token_id
        if token_id == ORIGINAL_EOS_TOKEN_ID:
            return self.eos_token_id
        return token_id

    def remap_original_lottie_token(self, token_id: int) -> int:
        token_id = self.remap_original_special_token(token_id)
        if ORIGINAL_BASE_VOCAB_SIZE <= token_id < ORIGINAL_VOCAB_SIZE:
            return token_id + self.shift
        return token_id

    def restore_opensource_token(self, token_id: int) -> int:
        if token_id == self.pad_token_id:
            return ORIGINAL_PAD_TOKEN_ID
        if token_id == self.bos_token_id:
            return ORIGINAL_BOS_TOKEN_ID
        if token_id == self.eos_token_id:
            return ORIGINAL_EOS_TOKEN_ID
        if self.is_lottie_token(token_id):
            return token_id - self.shift
        return token_id

    def remap_param_offset(self, command_idx: int, param_idx: int) -> int:
        original_offset = LottieTensor.get_param_offset(command_idx, param_idx)
        if original_offset == 0:
            return 0
        return original_offset + self.shift

