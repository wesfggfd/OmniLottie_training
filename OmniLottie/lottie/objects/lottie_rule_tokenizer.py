from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from transformers import AutoConfig, AutoTokenizer

from lottie.objects.animation import Animation
from lottie.objects.lottie_param import from_sequence
from lottie.objects.lottie_tokenize import LottieTensor


ORIGINAL_BASE_VOCAB_SIZE = 151643
ORIGINAL_VOCAB_SIZE = 192400
ORIGINAL_COMMAND_OFFSET = 151936
ORIGINAL_NUMBER_OFFSET = 173186
ORIGINAL_PAD_TOKEN_ID = 151643
ORIGINAL_BOS_TOKEN_ID = 192398
ORIGINAL_EOS_TOKEN_ID = 192399

SUPPORTED_LAYER_TYPES = {0, 1, 3, 4, 5}


@dataclass(frozen=True)
class TokenGroupSpec:
    count_index: int
    start_index: int
    max_tokens: int


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

    def command_token_id(self, command_idx: int) -> int:
        return self.command_offset + command_idx

    def count_token_id(self, count: int) -> int:
        return self.number_offset + count

    def remap_param_offset(self, command_idx: int, param_idx: int) -> int:
        original_offset = LottieTensor.get_param_offset(command_idx, param_idx)
        if original_offset == 0:
            return 0
        return original_offset + self.shift


class LottieSchemaFilter:
    @staticmethod
    def filter_animation(animation: Dict[str, Any]) -> Dict[str, Any]:
        filtered = dict(animation)
        filtered["layers"] = [
            layer for layer in animation.get("layers", []) if layer.get("ty") in SUPPORTED_LAYER_TYPES
        ]
        filtered["assets"] = [
            asset for asset in animation.get("assets", []) if LottieSchemaFilter._asset_is_supported(asset)
        ]
        return filtered

    @staticmethod
    def _asset_is_supported(asset: Dict[str, Any]) -> bool:
        if "layers" not in asset:
            return True
        return all(layer.get("ty") in SUPPORTED_LAYER_TYPES for layer in asset.get("layers", []))


class LottieRuleTokenizer:
    def __init__(self, base_model_name: str = "Qwen/Qwen3.5-9B"):
        self.base_model_name = base_model_name
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        base_vocab_size = getattr(getattr(base_config, "text_config", base_config), "vocab_size")
        self.vocab = LottieVocabLayout(base_vocab_size=base_vocab_size)
        LottieTensor.init_tokenizer(base_model_name)

    @property
    def bos_token_id(self) -> int:
        return self.vocab.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.vocab.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.vocab.pad_token_id

    @staticmethod
    def tokenizer_command_specs() -> Dict[int, Dict[str, Any]]:
        return {
            LottieTensor.CMD_FONT: {
                "regular": [LottieTensor.Index.Font.ASCENT],
                "token_groups": [
                    TokenGroupSpec(LottieTensor.Index.Font.FAMILY_TOKEN_COUNT, LottieTensor.Index.Font.FAMILY_TOKEN_0, 10),
                    TokenGroupSpec(LottieTensor.Index.Font.STYLE_TOKEN_COUNT, LottieTensor.Index.Font.STYLE_TOKEN_0, 10),
                ],
            },
            LottieTensor.CMD_CHAR: {
                "regular": [LottieTensor.Index.Char.SIZE, LottieTensor.Index.Char.W],
                "token_groups": [
                    TokenGroupSpec(LottieTensor.Index.Char.CH_TOKEN_COUNT, LottieTensor.Index.Char.CH_TOKEN_0, 10),
                    TokenGroupSpec(LottieTensor.Index.Char.STYLE_TOKEN_COUNT, LottieTensor.Index.Char.STYLE_TOKEN_0, 10),
                    TokenGroupSpec(LottieTensor.Index.Char.FAMILY_TOKEN_COUNT, LottieTensor.Index.Char.FAMILY_TOKEN_0, 10),
                ],
            },
            LottieTensor.CMD_ASSET: {
                "regular": [LottieTensor.Index.Asset.FR],
                "token_groups": [
                    TokenGroupSpec(LottieTensor.Index.Asset.ID_TOKEN_COUNT, LottieTensor.Index.Asset.ID_TOKEN_0, 10),
                ],
            },
            LottieTensor.CMD_REFERENCE_ID: {
                "regular": [],
                "token_groups": [
                    TokenGroupSpec(LottieTensor.Index.ReferenceId.ID_TOKEN_COUNT, LottieTensor.Index.ReferenceId.ID_TOKEN_0, 10),
                ],
            },
            LottieTensor.CMD_TEXT_KEYFRAME: {
                "regular": list(range(LottieTensor.Index.TextKeyframe.FONT_FAMILY_TOKENS_START)),
                "token_groups": [
                    TokenGroupSpec(
                        LottieTensor.Index.TextKeyframe.FONT_FAMILY_TOKEN_COUNT,
                        LottieTensor.Index.TextKeyframe.FONT_FAMILY_TOKENS_START,
                        10,
                    ),
                    TokenGroupSpec(
                        LottieTensor.Index.TextKeyframe.TEXT_TOKEN_COUNT,
                        LottieTensor.Index.TextKeyframe.TEXT_TOKENS_START,
                        15,
                    ),
                ],
            },
        }

    def encode_text_field(self, text: str, max_tokens: int = 15) -> List[int]:
        return self.base_tokenizer.encode(text, add_special_tokens=False)[:max_tokens]

    def encode_sequence_text(self, sequence_text: str, max_length: int | None = None) -> List[int]:
        parsed = from_sequence(sequence_text)
        normalized = Animation.load(parsed).to_dict()
        return self._encode_animation_dict(normalized, max_length=max_length)

    def encode_lottie_json(self, animation: Dict[str, Any] | str, max_length: int | None = None) -> List[int]:
        if isinstance(animation, str):
            stripped = animation.lstrip()
            if stripped.startswith("("):
                return self.encode_sequence_text(animation, max_length=max_length)
            animation = json.loads(animation)

        filtered = LottieSchemaFilter.filter_animation(animation)
        if "sequence_text" in filtered:
            return self.encode_sequence_text(filtered["sequence_text"], max_length=max_length)
        normalized = Animation.load(filtered).to_dict()
        return self._encode_animation_dict(normalized, max_length=max_length)

    def decode_token_ids(self, token_ids: Sequence[int]) -> LottieTensor:
        LottieTensor.init_tokenizer(self.base_model_name)
        return LottieTensor.from_list(list(token_ids))

    def wrap_target(self, token_ids: Sequence[int]) -> List[int]:
        return [self.bos_token_id, *token_ids, self.eos_token_id]

    def token_ids_to_sequence(self, token_ids: Sequence[int]) -> str:
        return self.decode_token_ids(token_ids).to_sequence()

    def token_ids_to_lottie_json(self, token_ids: Sequence[int]) -> Dict[str, Any]:
        sequence_text = self.token_ids_to_sequence(token_ids)
        return from_sequence(sequence_text)

    def _encode_numeric(self, command_idx: int, param_idx: int, value: float) -> int:
        offset = self.vocab.remap_param_offset(command_idx, param_idx)
        if offset == 0:
            return int(round(value))
        return int(round(value)) + offset

    def _encode_animation_dict(self, animation: Dict[str, Any], max_length: int | None = None) -> List[int]:
        tokens: List[int] = []
        self._append_simple_command(
            tokens,
            LottieTensor.CMD_ANIMATION,
            {
                LottieTensor.Index.Animation.FR: animation.get("fr", 30),
                LottieTensor.Index.Animation.IP: animation.get("ip", 0),
                LottieTensor.Index.Animation.OP: animation.get("op", 60),
                LottieTensor.Index.Animation.W: animation.get("w", 512),
                LottieTensor.Index.Animation.H: animation.get("h", 512),
                LottieTensor.Index.Animation.DDD: animation.get("ddd", 0),
            },
        )

        for asset in animation.get("assets", []):
            if "layers" not in asset:
                continue
            self._append_token_group_command(
                tokens,
                LottieTensor.CMD_ASSET,
                {LottieTensor.Index.Asset.FR: asset.get("fr", 0)},
                [(asset.get("id", ""), 10)],
            )
            for layer in asset.get("layers", []):
                self._append_layer(tokens, layer)
            self._append_no_param_command(tokens, LottieTensor.CMD_ASSET_END)

        for layer in animation.get("layers", []):
            self._append_layer(tokens, layer)

        if max_length is not None:
            return tokens[:max_length]
        return tokens

    def _append_layer(self, tokens: List[int], layer: Dict[str, Any]) -> None:
        layer_type = layer.get("ty")
        if layer_type == 4:
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_LAYER,
                {
                    LottieTensor.Index.Layer.INDEX: layer.get("ind", 0),
                    LottieTensor.Index.Layer.IN_POINT: layer.get("ip", 0),
                    LottieTensor.Index.Layer.OUT_POINT: layer.get("op", 60),
                    LottieTensor.Index.Layer.START_TIME: layer.get("st", 0),
                    LottieTensor.Index.Layer.DDD: layer.get("ddd", 0),
                    LottieTensor.Index.Layer.HD: self._bool(layer.get("hd", False)),
                    LottieTensor.Index.Layer.CP: self._bool(layer.get("cp", False)),
                    LottieTensor.Index.Layer.HAS_MASK: self._bool(layer.get("hasMask", False)),
                    LottieTensor.Index.Layer.AO: layer.get("ao", 0),
                    LottieTensor.Index.Layer.TT: layer.get("tt", 0),
                    LottieTensor.Index.Layer.TP: layer.get("tp", 0),
                    LottieTensor.Index.Layer.TD: layer.get("td", 0),
                    LottieTensor.Index.Layer.CT: layer.get("ct", 0),
                },
            )
            self._append_parent(tokens, layer)
            self._append_transform(tokens, layer.get("ks", {}))
            for shape in layer.get("shapes", []):
                self._append_shape(tokens, shape)
            self._append_no_param_command(tokens, LottieTensor.CMD_LAYER_END)
            return

        if layer_type == 3:
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_NULL_LAYER,
                {
                    LottieTensor.Index.NullLayer.INDEX: layer.get("ind", 0),
                    LottieTensor.Index.NullLayer.IN_POINT: layer.get("ip", 0),
                    LottieTensor.Index.NullLayer.OUT_POINT: layer.get("op", 60),
                    LottieTensor.Index.NullLayer.START_TIME: layer.get("st", 0),
                    LottieTensor.Index.NullLayer.CT: layer.get("ct", 0),
                    LottieTensor.Index.NullLayer.HD: self._bool(layer.get("hd", False)),
                    LottieTensor.Index.NullLayer.HAS_MASK: self._bool(layer.get("hasMask", False)),
                    LottieTensor.Index.NullLayer.AO: layer.get("ao", 0),
                    LottieTensor.Index.NullLayer.TT: layer.get("tt", 0),
                    LottieTensor.Index.NullLayer.TP: layer.get("tp", 0),
                    LottieTensor.Index.NullLayer.TD: layer.get("td", 0),
                    LottieTensor.Index.NullLayer.CP: self._bool(layer.get("cp", False)),
                },
            )
            self._append_parent(tokens, layer)
            self._append_transform(tokens, layer.get("ks", {}))
            self._append_no_param_command(tokens, LottieTensor.CMD_NULL_LAYER_END)
            return

        if layer_type == 0:
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_PRECOMP_LAYER,
                {
                    LottieTensor.Index.PrecompLayer.INDEX: layer.get("ind", 0),
                    LottieTensor.Index.PrecompLayer.IN_POINT: layer.get("ip", 0),
                    LottieTensor.Index.PrecompLayer.OUT_POINT: layer.get("op", 60),
                    LottieTensor.Index.PrecompLayer.START_TIME: layer.get("st", 0),
                    LottieTensor.Index.PrecompLayer.W: layer.get("w", 512),
                    LottieTensor.Index.PrecompLayer.H: layer.get("h", 512),
                    LottieTensor.Index.PrecompLayer.CT: layer.get("ct", 0),
                    LottieTensor.Index.PrecompLayer.HAS_MASK: self._bool(layer.get("hasMask", False)),
                    LottieTensor.Index.PrecompLayer.AO: layer.get("ao", 0),
                    LottieTensor.Index.PrecompLayer.TT: layer.get("tt", 0),
                    LottieTensor.Index.PrecompLayer.TP: layer.get("tp", 0),
                    LottieTensor.Index.PrecompLayer.TD: layer.get("td", 0),
                    LottieTensor.Index.PrecompLayer.DDD: layer.get("ddd", 0),
                    LottieTensor.Index.PrecompLayer.HD: self._bool(layer.get("hd", False)),
                    LottieTensor.Index.PrecompLayer.CP: self._bool(layer.get("cp", False)),
                },
            )
            self._append_parent(tokens, layer)
            if layer.get("refId"):
                self._append_token_group_command(tokens, LottieTensor.CMD_REFERENCE_ID, {}, [(layer["refId"], 10)])
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_DIMENSIONS,
                {
                    LottieTensor.Index.Dimensions.WIDTH: layer.get("w", 512),
                    LottieTensor.Index.Dimensions.HEIGHT: layer.get("h", 512),
                },
            )
            self._append_transform(tokens, layer.get("ks", {}))
            self._append_no_param_command(tokens, LottieTensor.CMD_PRECOMP_LAYER_END)
            return

        if layer_type == 1:
            rgba = self._color_to_255(layer.get("sc", "#000000"))
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_SOLID_LAYER,
                {
                    LottieTensor.Index.SolidLayer.INDEX: layer.get("ind", 0),
                    LottieTensor.Index.SolidLayer.IN_POINT: layer.get("ip", 0),
                    LottieTensor.Index.SolidLayer.OUT_POINT: layer.get("op", 60),
                    LottieTensor.Index.SolidLayer.START_TIME: layer.get("st", 0),
                    LottieTensor.Index.SolidLayer.WIDTH: layer.get("sw", 512),
                    LottieTensor.Index.SolidLayer.HEIGHT: layer.get("sh", 512),
                    LottieTensor.Index.SolidLayer.HAS_MASK: self._bool(layer.get("hasMask", False)),
                    LottieTensor.Index.SolidLayer.COLOR_R: rgba[0],
                    LottieTensor.Index.SolidLayer.COLOR_G: rgba[1],
                    LottieTensor.Index.SolidLayer.COLOR_B: rgba[2],
                    LottieTensor.Index.SolidLayer.COLOR_A: rgba[3],
                },
            )
            self._append_parent(tokens, layer)
            self._append_transform(tokens, layer.get("ks", {}))
            self._append_no_param_command(tokens, LottieTensor.CMD_SOLID_LAYER_END)
            return

        if layer_type == 5:
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_TEXT_LAYER,
                {
                    LottieTensor.Index.TextLayer.INDEX: layer.get("ind", 0),
                    LottieTensor.Index.TextLayer.IN_POINT: layer.get("ip", 0),
                    LottieTensor.Index.TextLayer.OUT_POINT: layer.get("op", 60),
                    LottieTensor.Index.TextLayer.START_TIME: layer.get("st", 0),
                    LottieTensor.Index.TextLayer.HAS_MASK: self._bool(layer.get("hasMask", False)),
                },
            )
            self._append_parent(tokens, layer)
            self._append_transform(tokens, layer.get("ks", {}))
            self._append_text_data(tokens, layer.get("t", {}))
            self._append_no_param_command(tokens, LottieTensor.CMD_TEXT_LAYER_END)

    def _append_text_data(self, tokens: List[int], text_data: Dict[str, Any]) -> None:
        self._append_no_param_command(tokens, LottieTensor.CMD_TEXT_DATA)
        keyframes = ((text_data or {}).get("d") or {}).get("k") or []
        self._append_no_param_command(tokens, LottieTensor.CMD_TEXT_KEYFRAMES)
        for keyframe in keyframes:
            doc = keyframe.get("s", {})
            fill_color = self._color_array_to_255(doc.get("fc", [0, 0, 0]))
            stroke_color = self._color_array_to_255(doc.get("sc", [0, 0, 0]))
            token_groups = [
                (doc.get("f", ""), 10),
                (doc.get("t", ""), 15),
            ]
            self._append_token_group_command(
                tokens,
                LottieTensor.CMD_TEXT_KEYFRAME,
                {
                    LottieTensor.Index.TextKeyframe.T: keyframe.get("t", 0),
                    LottieTensor.Index.TextKeyframe.STROKE_WIDTH: doc.get("sw", 0),
                    LottieTensor.Index.TextKeyframe.OFFSET: self._bool(doc.get("of", False)),
                    LottieTensor.Index.TextKeyframe.WRAP_POSITION_X: (doc.get("ps") or [0, 0])[0],
                    LottieTensor.Index.TextKeyframe.WRAP_POSITION_Y: (doc.get("ps") or [0, 0])[-1],
                    LottieTensor.Index.TextKeyframe.WRAP_SIZE_X: (doc.get("sz") or [0, 0])[0],
                    LottieTensor.Index.TextKeyframe.WRAP_SIZE_Y: (doc.get("sz") or [0, 0])[-1],
                    LottieTensor.Index.TextKeyframe.FONT_SIZE: doc.get("s", 0),
                    LottieTensor.Index.TextKeyframe.CA: doc.get("ca", 0),
                    LottieTensor.Index.TextKeyframe.JUSTIFY: doc.get("j", 0),
                    LottieTensor.Index.TextKeyframe.TRACKING: doc.get("tr", 0),
                    LottieTensor.Index.TextKeyframe.LINE_HEIGHT: doc.get("lh", 0),
                    LottieTensor.Index.TextKeyframe.LETTER_SPACING: doc.get("ls", 0),
                    LottieTensor.Index.TextKeyframe.FILL_COLOR_R: fill_color[0],
                    LottieTensor.Index.TextKeyframe.FILL_COLOR_G: fill_color[1],
                    LottieTensor.Index.TextKeyframe.FILL_COLOR_B: fill_color[2],
                    LottieTensor.Index.TextKeyframe.STROKE_COLOR_R: stroke_color[0],
                    LottieTensor.Index.TextKeyframe.STROKE_COLOR_G: stroke_color[1],
                    LottieTensor.Index.TextKeyframe.STROKE_COLOR_B: stroke_color[2],
                    LottieTensor.Index.TextKeyframe.HAS_STROKE_COLOR: self._bool("sc" in doc),
                },
                token_groups,
            )
        self._append_no_param_command(tokens, LottieTensor.CMD_TEXT_KEYFRAMES_END)
        self._append_no_param_command(tokens, LottieTensor.CMD_TEXT_DATA_END)

    def _append_parent(self, tokens: List[int], node: Dict[str, Any]) -> None:
        if node.get("parent") is not None:
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_PARENT,
                {LottieTensor.Index.Parent.PARENT_INDEX: node["parent"]},
            )

    def _append_transform(self, tokens: List[int], transform: Dict[str, Any]) -> None:
        self._append_no_param_command(tokens, LottieTensor.CMD_TRANSFORM)
        self._append_transform_prop(tokens, LottieTensor.CMD_POSITION, LottieTensor.CMD_POSITION_END, transform.get("p"), dims=3)
        self._append_transform_prop(tokens, LottieTensor.CMD_SCALE, LottieTensor.CMD_SCALE_END, transform.get("s"), dims=3)
        self._append_transform_prop(tokens, LottieTensor.CMD_ROTATION, LottieTensor.CMD_ROTATION_END, transform.get("r"), dims=1)
        self._append_transform_prop(tokens, LottieTensor.CMD_OPACITY, LottieTensor.CMD_OPACITY_END, transform.get("o"), dims=1)
        self._append_transform_prop(tokens, LottieTensor.CMD_ANCHOR, LottieTensor.CMD_ANCHOR_END, transform.get("a"), dims=3)
        self._append_no_param_command(tokens, LottieTensor.CMD_TRANSFORM_END)

    def _append_transform_prop(
        self,
        tokens: List[int],
        cmd_idx: int,
        end_cmd_idx: int,
        prop: Dict[str, Any] | None,
        dims: int,
    ) -> None:
        if not prop:
            return
        values = self._first_prop_values(prop, dims, defaults=[0.0] * dims)
        params = {LottieTensor.Index.Transform.ANIMATED: self._bool(prop.get("a", 0) == 1)}
        if values:
            params[LottieTensor.Index.Transform.X] = values[0]
        if len(values) > 1:
            params[LottieTensor.Index.Transform.Y] = values[1]
        if len(values) > 2:
            params[LottieTensor.Index.Transform.Z] = values[2]
        self._append_simple_command(tokens, cmd_idx, params)
        if prop.get("a", 0) == 1:
            for keyframe in prop.get("k", []):
                self._append_keyframe(tokens, keyframe, dims=dims)
            self._append_no_param_command(tokens, end_cmd_idx)

    def _append_keyframe(self, tokens: List[int], keyframe: Dict[str, Any], dims: int) -> None:
        params: Dict[int, float] = {LottieTensor.Index.Keyframe.T: keyframe.get("t", 0)}
        start = keyframe.get("s", [0.0] * dims)
        if not isinstance(start, list):
            start = [start]
        for idx, value in enumerate(start[:3], start=1):
            params[getattr(LottieTensor.Index.Keyframe, f"S{idx}")] = value

        end = keyframe.get("e") or []
        if isinstance(end, list):
            for idx, value in enumerate(end[:3], start=1):
                params[getattr(LottieTensor.Index.Keyframe, f"E{idx}")] = value

        for tangent_key, prefix in (("i", "I"), ("o", "O")):
            tangent = keyframe.get(tangent_key)
            if not tangent:
                continue
            x_vals = tangent.get("x")
            y_vals = tangent.get("y")
            if not isinstance(x_vals, list):
                x_vals = [x_vals] if x_vals is not None else []
            if not isinstance(y_vals, list):
                y_vals = [y_vals] if y_vals is not None else []
            if x_vals:
                params[LottieTensor.Index.Keyframe.__dict__[f"{prefix}_X"]] = x_vals[0]
            if y_vals:
                params[LottieTensor.Index.Keyframe.__dict__[f"{prefix}_Y"]] = y_vals[0]
            if len(x_vals) > 1:
                params[LottieTensor.Index.Keyframe.__dict__[f"{prefix}_X2"]] = x_vals[1]
            if len(x_vals) > 2:
                params[LottieTensor.Index.Keyframe.__dict__[f"{prefix}_X3"]] = x_vals[2]
            if len(y_vals) > 1:
                params[LottieTensor.Index.Keyframe.__dict__[f"{prefix}_Y2"]] = y_vals[1]
            if len(y_vals) > 2:
                params[LottieTensor.Index.Keyframe.__dict__[f"{prefix}_Y3"]] = y_vals[2]

        for idx, value in enumerate((keyframe.get("to") or [])[:3], start=1):
            params[getattr(LottieTensor.Index.Keyframe, f"TO{idx}")] = value
        for idx, value in enumerate((keyframe.get("ti") or [])[:3], start=1):
            params[getattr(LottieTensor.Index.Keyframe, f"TI{idx}")] = value
        if keyframe.get("h") is not None:
            params[LottieTensor.Index.Keyframe.H_FLAG] = keyframe["h"]
        self._append_simple_command(tokens, LottieTensor.CMD_KEYFRAME, params)

    def _append_shape(self, tokens: List[int], shape: Dict[str, Any]) -> None:
        shape_type = shape.get("ty")
        if shape_type == "gr":
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_GROUP,
                {
                    LottieTensor.Index.Group.IX: shape.get("ix", 1),
                    LottieTensor.Index.Group.CIX: shape.get("cix", 2),
                    LottieTensor.Index.Group.BM: shape.get("bm", 0),
                    LottieTensor.Index.Group.HD: self._bool(shape.get("hd", False)),
                    LottieTensor.Index.Group.NP: shape.get("np", len(shape.get("it", []))),
                },
            )
            for child in shape.get("it", []):
                self._append_shape(tokens, child)
            self._append_no_param_command(tokens, LottieTensor.CMD_GROUP_END)
            return

        if shape_type == "tr":
            pos = self._first_prop_values(shape.get("p"), 2, [0, 0])
            scale = self._first_prop_values(shape.get("s"), 2, [100, 100])
            anchor = self._first_prop_values(shape.get("a"), 2, [0, 0])
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_TRANSFORM_SHAPE,
                {
                    LottieTensor.Index.TransformShape.POSITION_X: pos[0],
                    LottieTensor.Index.TransformShape.POSITION_Y: pos[1],
                    LottieTensor.Index.TransformShape.SCALE_X: scale[0],
                    LottieTensor.Index.TransformShape.SCALE_Y: scale[1],
                    LottieTensor.Index.TransformShape.ROTATION: self._first_scalar_prop(shape.get("r"), 0),
                    LottieTensor.Index.TransformShape.OPACITY: self._first_scalar_prop(shape.get("o"), 100),
                    LottieTensor.Index.TransformShape.ANCHOR_X: anchor[0],
                    LottieTensor.Index.TransformShape.ANCHOR_Y: anchor[1],
                    LottieTensor.Index.TransformShape.SKEW: self._first_scalar_prop(shape.get("sk"), 0),
                    LottieTensor.Index.TransformShape.SKEW_AXIS: self._first_scalar_prop(shape.get("sa"), 0),
                    LottieTensor.Index.TransformShape.HD: self._bool(shape.get("hd", False)),
                },
            )
            return

        if shape_type == "sh":
            bezier = self._first_bezier(shape.get("ks"))
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_PATH,
                {
                    LottieTensor.Index.Path.IX: shape.get("ix", 1),
                    LottieTensor.Index.Path.IND: shape.get("ind", 0),
                    LottieTensor.Index.Path.KS_IX: 2,
                    LottieTensor.Index.Path.CLOSED: self._bool(bezier.get("c", True)),
                    LottieTensor.Index.Path.HD: self._bool(shape.get("hd", False)),
                    LottieTensor.Index.Path.ANIMATED: self._bool(shape.get("ks", {}).get("a", 0) == 1),
                },
            )
            for vertex, in_tan, out_tan in zip(bezier.get("v", []), bezier.get("i", []), bezier.get("o", [])):
                self._append_simple_command(
                    tokens,
                    LottieTensor.CMD_POINT,
                    {
                        LottieTensor.Index.Point.X: vertex[0],
                        LottieTensor.Index.Point.Y: vertex[1],
                        LottieTensor.Index.Point.IN_X: in_tan[0],
                        LottieTensor.Index.Point.IN_Y: in_tan[1],
                        LottieTensor.Index.Point.OUT_X: out_tan[0],
                        LottieTensor.Index.Point.OUT_Y: out_tan[1],
                    },
                )
            self._append_no_param_command(tokens, LottieTensor.CMD_PATH_END)
            return

        if shape_type == "fl":
            color = self._color_array_to_255(self._first_prop_values(shape.get("c"), 4, [0, 0, 0, 1]))
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_FILL,
                {
                    LottieTensor.Index.Fill.R: color[0],
                    LottieTensor.Index.Fill.G: color[1],
                    LottieTensor.Index.Fill.B: color[2],
                    LottieTensor.Index.Fill.COLOR_DIM: len(color),
                    LottieTensor.Index.Fill.HAS_C_A: self._bool(len(color) > 3),
                    LottieTensor.Index.Fill.HAS_C_IX: 0,
                    LottieTensor.Index.Fill.C_IX: shape.get("c", {}).get("ix", 0),
                    LottieTensor.Index.Fill.BM: shape.get("bm", 0),
                    LottieTensor.Index.Fill.FILL_RULE: shape.get("r", 1),
                    LottieTensor.Index.Fill.OPACITY: self._first_scalar_prop(shape.get("o"), 100),
                    LottieTensor.Index.Fill.COLOR_ANIMATED: self._bool(shape.get("c", {}).get("a", 0) == 1),
                    LottieTensor.Index.Fill.OPACITY_ANIMATED: self._bool(shape.get("o", {}).get("a", 0) == 1),
                    LottieTensor.Index.Fill.HAS_O_A: 0,
                    LottieTensor.Index.Fill.HAS_O_IX: 0,
                    LottieTensor.Index.Fill.O_IX: shape.get("o", {}).get("ix", 0),
                },
            )
            return

        if shape_type == "st":
            color = self._color_array_to_255(self._first_prop_values(shape.get("c"), 4, [0, 0, 0, 1]))
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_STROKE,
                {
                    LottieTensor.Index.Stroke.R: color[0],
                    LottieTensor.Index.Stroke.G: color[1],
                    LottieTensor.Index.Stroke.B: color[2],
                    LottieTensor.Index.Stroke.COLOR_DIM: len(color),
                    LottieTensor.Index.Stroke.HAS_C_A: self._bool(len(color) > 3),
                    LottieTensor.Index.Stroke.HAS_C_IX: 0,
                    LottieTensor.Index.Stroke.C_IX: shape.get("c", {}).get("ix", 0),
                    LottieTensor.Index.Stroke.BM: shape.get("bm", 0),
                    LottieTensor.Index.Stroke.LC: shape.get("lc", 2),
                    LottieTensor.Index.Stroke.LJ: shape.get("lj", 2),
                    LottieTensor.Index.Stroke.ML: shape.get("ml", 4),
                    LottieTensor.Index.Stroke.WIDTH_ANIMATED: self._bool(shape.get("w", {}).get("a", 0) == 1),
                    LottieTensor.Index.Stroke.COLOR_ANIMATED: self._bool(shape.get("c", {}).get("a", 0) == 1),
                    LottieTensor.Index.Stroke.A: color[3] if len(color) > 3 else 255,
                },
            )
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_WIDTH,
                {LottieTensor.Index.SingleValue.VALUE: self._first_scalar_prop(shape.get("w"), 1) * 10},
            )
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_OPACITY,
                {
                    LottieTensor.Index.Transform.ANIMATED: self._bool(shape.get("o", {}).get("a", 0) == 1),
                    LottieTensor.Index.Transform.X: self._first_scalar_prop(shape.get("o"), 100),
                },
            )
            return

        if shape_type == "rc":
            pos = self._first_prop_values(shape.get("p"), 2, [0, 0])
            size = self._first_prop_values(shape.get("s"), 2, [0, 0])
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_RECT,
                {
                    LottieTensor.Index.Rect.HD: self._bool(shape.get("hd", False)),
                    LottieTensor.Index.Rect.D: shape.get("d", 1),
                    LottieTensor.Index.Rect.POSITION_X: pos[0],
                    LottieTensor.Index.Rect.POSITION_Y: pos[1],
                    LottieTensor.Index.Rect.SIZE_X: size[0],
                    LottieTensor.Index.Rect.SIZE_Y: size[1],
                    LottieTensor.Index.Rect.ROUNDED: self._first_scalar_prop(shape.get("r"), 0),
                    LottieTensor.Index.Rect.IX: shape.get("ix", 1),
                },
            )
            self._append_no_param_command(tokens, LottieTensor.CMD_RECT_END)
            return

        if shape_type == "el":
            pos = self._first_prop_values(shape.get("p"), 2, [0, 0])
            size = self._first_prop_values(shape.get("s"), 2, [0, 0])
            self._append_simple_command(
                tokens,
                LottieTensor.CMD_ELLIPSE,
                {
                    LottieTensor.Index.Ellipse.POSITION_X: pos[0],
                    LottieTensor.Index.Ellipse.POSITION_Y: pos[1],
                    LottieTensor.Index.Ellipse.SIZE_X: size[0],
                    LottieTensor.Index.Ellipse.SIZE_Y: size[1],
                },
            )
            self._append_no_param_command(tokens, LottieTensor.CMD_ELLIPSE_END)

    def _append_simple_command(self, tokens: List[int], command_idx: int, param_values: Dict[int, float]) -> None:
        tokens.append(self.vocab.command_token_id(command_idx))
        for param_idx in LottieTensor.get_command_param_indices(command_idx):
            value = param_values.get(param_idx, LottieTensor.PAD_VAL)
            if value == LottieTensor.PAD_VAL:
                continue
            tokens.append(self._encode_numeric(command_idx, param_idx, float(value)))

    def _append_token_group_command(
        self,
        tokens: List[int],
        command_idx: int,
        regular_values: Dict[int, float],
        text_groups: List[tuple[str, int]],
    ) -> None:
        tokens.append(self.vocab.command_token_id(command_idx))
        spec = self.tokenizer_command_specs()[command_idx]
        for param_idx in spec["regular"]:
            value = regular_values.get(param_idx, LottieTensor.PAD_VAL)
            if value == LottieTensor.PAD_VAL:
                continue
            tokens.append(self._encode_numeric(command_idx, param_idx, float(value)))
        for group, (text, max_tokens) in zip(spec["token_groups"], text_groups):
            text_token_ids = self.encode_text_field(text or "", max_tokens=max_tokens)
            tokens.append(self.vocab.count_token_id(len(text_token_ids)))
            tokens.extend(text_token_ids)

    def _append_no_param_command(self, tokens: List[int], command_idx: int) -> None:
        tokens.append(self.vocab.command_token_id(command_idx))

    @staticmethod
    def _bool(value: Any) -> int:
        return 1 if bool(value) else 0

    @staticmethod
    def _first_scalar_prop(prop: Dict[str, Any] | None, default: float) -> float:
        if not prop:
            return default
        value = prop.get("k", default)
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                start = value[0].get("s", [default])
                return float(start[0] if isinstance(start, list) else start)
            return float(value[0])
        return float(value)

    @staticmethod
    def _first_prop_values(prop: Dict[str, Any] | None, dims: int, defaults: List[float]) -> List[float]:
        if not prop:
            return defaults
        value = prop.get("k", defaults)
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                start = value[0].get("s", defaults)
                if isinstance(start, list):
                    return [float(v) for v in start[:dims]] + defaults[len(start[:dims]) : dims]
                return [float(start)] + defaults[1:dims]
            return [float(v) for v in value[:dims]] + defaults[len(value[:dims]) : dims]
        return defaults

    @staticmethod
    def _first_bezier(prop: Dict[str, Any] | None) -> Dict[str, Any]:
        if not prop:
            return {"c": True, "v": [], "i": [], "o": []}
        value = prop.get("k", {})
        if isinstance(value, list) and value and isinstance(value[0], dict):
            start = value[0].get("s", [{}])
            if start and isinstance(start[0], dict):
                return start[0]
        if isinstance(value, dict):
            return value
        return {"c": True, "v": [], "i": [], "o": []}

    @staticmethod
    def _color_array_to_255(values: List[float]) -> List[int]:
        rgba = list(values)[:4]
        while len(rgba) < 4:
            rgba.append(1.0 if len(rgba) == 3 else 0.0)
        return [int(round(max(0.0, min(1.0, channel)) * 255)) for channel in rgba]

    @staticmethod
    def _color_to_255(value: Any) -> List[int]:
        if isinstance(value, str) and value.startswith("#"):
            hex_value = value[1:]
            if len(hex_value) == 6:
                return [int(hex_value[i : i + 2], 16) for i in (0, 2, 4)] + [255]
            if len(hex_value) == 8:
                return [int(hex_value[i : i + 2], 16) for i in (0, 2, 4, 6)]
        return [0, 0, 0, 255]
