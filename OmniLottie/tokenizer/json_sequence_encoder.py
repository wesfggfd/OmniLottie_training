from __future__ import annotations

import json
from typing import Any, Dict, List

from lottie.objects.animation import Animation


SUPPORTED_LAYER_TYPES = {0, 1, 3, 4, 5}


def animation_to_sequence_text(animation: Dict[str, Any]) -> str:
    normalized = Animation.load(animation).to_dict()
    lines: List[str] = []

    lines.append(
        _tag(
            "animation",
            v=normalized.get("v", "5.5.2"),
            fr=normalized.get("fr", 30),
            ip=normalized.get("ip", 0),
            op=normalized.get("op", 60),
            w=normalized.get("w", 512),
            h=normalized.get("h", 512),
            ddd=normalized.get("ddd", 0),
        )
    )

    fonts = normalized.get("fonts", {}) or {}
    if fonts.get("list"):
        lines.append("(fonts)")
        for font in fonts["list"]:
            lines.append(
                _tag(
                    "font",
                    family=font.get("fFamily", ""),
                    style=font.get("fStyle", "Regular"),
                    name=font.get("fName", ""),
                    ascent=font.get("ascent", 0),
                )
            )
        lines.append("(/fonts)")

    if normalized.get("chars"):
        lines.append("(chars)")
        for char in normalized["chars"]:
            lines.append(
                _tag(
                    "char",
                    ch=char.get("ch", ""),
                    size=char.get("size", 0),
                    style=char.get("style", ""),
                    w=char.get("w", 0),
                    family=char.get("fFamily", ""),
                )
            )
            data = (char.get("data") or {}).get("shapes") or []
            if data:
                lines.append("(char_shapes)")
                for shape in data:
                    lines.extend(_serialize_shape(shape))
                lines.append("(/char_shapes)")
            lines.append("(/char)")
        lines.append("(/chars)")

    for asset in normalized.get("assets", []):
        if "layers" in asset:
            lines.append(_tag("asset", id=asset.get("id", ""), fr=asset.get("fr", 0)))
            for layer in asset.get("layers", []):
                if layer.get("ty") in SUPPORTED_LAYER_TYPES:
                    lines.extend(_serialize_layer(layer))
            lines.append("(/asset)")

    for layer in normalized.get("layers", []):
        if layer.get("ty") in SUPPORTED_LAYER_TYPES:
            lines.extend(_serialize_layer(layer))

    return "\n".join(lines)


def _serialize_layer(layer: Dict[str, Any]) -> List[str]:
    ty = layer.get("ty")
    if ty == 4:
        return _serialize_shape_layer(layer)
    if ty == 3:
        return _serialize_null_layer(layer)
    if ty == 0:
        return _serialize_precomp_layer(layer)
    if ty == 1:
        return _serialize_solid_layer(layer)
    if ty == 5:
        return _serialize_text_layer(layer)
    return []


def _serialize_shape_layer(layer: Dict[str, Any]) -> List[str]:
    lines = [
        _tag(
            "layer",
            index=layer.get("ind", 0),
            name=layer.get("nm", "Layer"),
            in_point=layer.get("ip", 0),
            out_point=layer.get("op", 60),
            start_time=layer.get("st", 0),
            ddd=layer.get("ddd", 0),
            hd=layer.get("hd", False),
            ao=layer.get("ao", 0),
            hasMask=layer.get("hasMask", False),
            tt=layer.get("tt"),
            tp=layer.get("tp"),
            td=layer.get("td"),
            ct=layer.get("ct"),
        )
    ]
    if layer.get("parent") is not None:
        lines.append(f"(parent {layer['parent']})")
    lines.extend(_serialize_transform(layer.get("ks", {})))
    for shape in layer.get("shapes", []):
        lines.extend(_serialize_shape(shape))
    lines.append("(/layer)")
    return lines


def _serialize_null_layer(layer: Dict[str, Any]) -> List[str]:
    lines = [
        _tag(
            "null_layer",
            index=layer.get("ind", 0),
            name=layer.get("nm", "Null Layer"),
            in_point=layer.get("ip", 0),
            out_point=layer.get("op", 60),
            start_time=layer.get("st", 0),
            hd=layer.get("hd", False),
            hasMask=layer.get("hasMask", False),
            ao=layer.get("ao", 0),
            tt=layer.get("tt"),
            tp=layer.get("tp"),
            td=layer.get("td"),
            ct=layer.get("ct"),
        )
    ]
    if layer.get("parent") is not None:
        lines.append(f"(parent {layer['parent']})")
    lines.extend(_serialize_transform(layer.get("ks", {})))
    lines.append("(/null_layer)")
    return lines


def _serialize_precomp_layer(layer: Dict[str, Any]) -> List[str]:
    lines = [
        _tag(
            "precomp_layer",
            index=layer.get("ind", 0),
            name=layer.get("nm", "PreComp Layer"),
            in_point=layer.get("ip", 0),
            out_point=layer.get("op", 60),
            start_time=layer.get("st", 0),
            w=layer.get("w", 512),
            h=layer.get("h", 512),
            hasMask=layer.get("hasMask", False),
            hd=layer.get("hd", False),
            cp=layer.get("cp", False),
            tt=layer.get("tt"),
            tp=layer.get("tp"),
            td=layer.get("td"),
        )
    ]
    if layer.get("parent") is not None:
        lines.append(f"(parent {layer['parent']})")
    if layer.get("refId"):
        lines.append(f'(reference_id "{_escape(layer["refId"])}")')
    lines.append(_tag("dimensions", width=layer.get("w", 512), height=layer.get("h", 512)))
    lines.extend(_serialize_transform(layer.get("ks", {})))
    lines.append("(/precomp_layer)")
    return lines


def _serialize_solid_layer(layer: Dict[str, Any]) -> List[str]:
    lines = [
        _tag(
            "solid_layer",
            index=layer.get("ind", 0),
            name=layer.get("nm", "Solid Layer"),
            in_point=layer.get("ip", 0),
            out_point=layer.get("op", 60),
            start_time=layer.get("st", 0),
            color=layer.get("sc", "#000000"),
            width=layer.get("sw", 512),
            height=layer.get("sh", 512),
            hasMask=layer.get("hasMask", False),
        )
    ]
    if layer.get("parent") is not None:
        lines.append(f"(parent {layer['parent']})")
    lines.extend(_serialize_transform(layer.get("ks", {})))
    lines.append("(/solid_layer)")
    return lines


def _serialize_text_layer(layer: Dict[str, Any]) -> List[str]:
    lines = [
        _tag(
            "text_layer",
            index=layer.get("ind", 0),
            name=layer.get("nm", "Text Layer"),
            in_point=layer.get("ip", 0),
            out_point=layer.get("op", 60),
            start_time=layer.get("st", 0),
            ct=layer.get("ct"),
            hasMask=layer.get("hasMask", False),
        )
    ]
    if layer.get("parent") is not None:
        lines.append(f"(parent {layer['parent']})")
    lines.extend(_serialize_transform(layer.get("ks", {})))

    text_data = layer.get("t", {}) or {}
    lines.append("(text_data)")
    if text_data.get("d", {}).get("k"):
        doc_json = json.dumps(text_data["d"]["k"], ensure_ascii=True).replace('"', '\\"')
        lines.append(f'(document_full "{doc_json}")')
    if text_data.get("p"):
        path_json = json.dumps(text_data["p"], ensure_ascii=True).replace('"', '\\"')
        lines.append(f'(path_option "{path_json}")')
    if text_data.get("m"):
        more_json = json.dumps(text_data["m"], ensure_ascii=True).replace('"', '\\"')
        lines.append(f'(more_options "{more_json}")')
    lines.append("(/text_data)")
    lines.append("(/text_layer)")
    return lines


def _serialize_transform(transform: Dict[str, Any]) -> List[str]:
    lines = ["(transform)"]
    mapping = {
        "p": "position",
        "s": "scale",
        "r": "rotation",
        "rz": "rotation",
        "o": "opacity",
        "a": "anchor",
    }
    for source_key, tag in mapping.items():
        if source_key in transform:
            lines.extend(_serialize_property_tag(tag, transform[source_key]))
    lines.append("(/transform)")
    return lines


def _serialize_shape(shape: Dict[str, Any]) -> List[str]:
    ty = shape.get("ty")
    if ty == "gr":
        return _serialize_group(shape)
    if ty == "sh":
        return _serialize_path(shape)
    if ty == "fl":
        return _serialize_fill(shape)
    if ty == "st":
        return _serialize_stroke(shape)
    if ty == "rc":
        return _serialize_rect(shape)
    if ty == "el":
        return _serialize_ellipse(shape)
    if ty == "tr":
        return _serialize_transform_shape(shape)
    return []


def _serialize_group(group: Dict[str, Any]) -> List[str]:
    lines = [
        _tag(
            "group",
            name=group.get("nm", ""),
            ix=group.get("ix"),
            cix=group.get("cix"),
            bm=group.get("bm"),
            hd=group.get("hd", False),
            np=group.get("np"),
        )
    ]
    for item in group.get("it", []):
        lines.extend(_serialize_shape(item))
    lines.append("(/group)")
    return lines


def _serialize_transform_shape(shape: Dict[str, Any]) -> List[str]:
    attrs = {
        "name": shape.get("nm", ""),
        "ix": shape.get("ix"),
        "hd": shape.get("hd", False),
        "position": _static_property_text(shape.get("p")),
        "scale": _static_property_text(shape.get("s")),
        "rotation": _first_scalar(shape.get("r")),
        "opacity": _first_scalar(shape.get("o")),
        "anchor": _static_property_text(shape.get("a")),
        "skew": _first_scalar(shape.get("sk")),
        "skew_axis": _first_scalar(shape.get("sa")),
    }
    return [_tag('"TransformShape"', **attrs)]


def _serialize_path(shape: Dict[str, Any]) -> List[str]:
    path_prop = shape.get("ks", {})
    is_animated = path_prop.get("a", 0) == 1
    lines = [
        _tag(
            "path",
            name=shape.get("nm", ""),
            ix=shape.get("ix"),
            d=shape.get("d", 1),
            ind=shape.get("ind", 1),
            hd=shape.get("hd", False),
            animated=is_animated if is_animated else None,
            closed=_path_closed(path_prop) if not is_animated else None,
        )
    ]
    if is_animated:
        for keyframe in path_prop.get("k", []):
            lines.append(_tag("keyframe", **_keyframe_attrs(keyframe)))
            bezier = (keyframe.get("s") or [{}])[0]
            lines.extend(_serialize_bezier(bezier))
            lines.append("(/keyframe)")
    else:
        lines.extend(_serialize_bezier(path_prop.get("k", {}), include_wrappers=False))
    lines.append("(/path)")
    return lines


def _serialize_bezier(bezier: Dict[str, Any], include_wrappers: bool = True) -> List[str]:
    lines: List[str] = []
    if include_wrappers:
        lines.append(_tag("bezier", closed=bezier.get("c", True)))
    points = zip(bezier.get("v", []), bezier.get("i", []), bezier.get("o", []))
    for vertex, in_tan, out_tan in points:
        lines.append(
            _tag(
                "point",
                x=vertex[0],
                y=vertex[1],
                in_x=in_tan[0],
                in_y=in_tan[1],
                out_x=out_tan[0],
                out_y=out_tan[1],
            )
        )
    if include_wrappers:
        lines.append("(/bezier)")
    return lines


def _serialize_fill(shape: Dict[str, Any]) -> List[str]:
    color = _first_vector(shape.get("c"), fallback=[0, 0, 0])
    opacity = _first_scalar(shape.get("o"), default=100)
    return [
        _tag(
            "fill",
            name=shape.get("nm", ""),
            ix=shape.get("ix"),
            bm=shape.get("bm", 0),
            r=color[0],
            g=color[1],
            b=color[2],
            color_dim=len(color),
            opacity=opacity,
            fill_rule=shape.get("r", 1),
        )
    ]


def _serialize_stroke(shape: Dict[str, Any]) -> List[str]:
    color = _first_vector(shape.get("c"), fallback=[0, 0, 0])
    lines = [
        _tag(
            "stroke",
            name=shape.get("nm", ""),
            ix=shape.get("ix"),
            bm=shape.get("bm", 0),
            r=color[0],
            g=color[1],
            b=color[2],
            color_dim=len(color),
            lc=shape.get("lc", 2),
            lj=shape.get("lj", 2),
            ml=shape.get("ml", 4),
        )
    ]
    lines.append(f"(width {_fmt(_first_scalar(shape.get('w'), default=1))})")
    lines.append(f"(opacity {_fmt(_first_scalar(shape.get('o'), default=100))})")
    lines.append("(/stroke)")
    return lines


def _serialize_rect(shape: Dict[str, Any]) -> List[str]:
    lines = [_tag("rect", name=shape.get("nm", ""), ix=shape.get("ix"), hd=shape.get("hd", False), d=shape.get("d", 1))]
    lines.extend(_serialize_property_tag("position", shape.get("p")))
    size_prop = shape.get("s")
    if size_prop and size_prop.get("a", 0) == 1:
        lines.extend(_serialize_property_tag("size", size_prop))
    else:
        lines.extend(_serialize_property_tag("rect_size", size_prop))
    lines.extend(_serialize_property_tag("rounded", shape.get("r")))
    lines.append("(/rect)")
    return lines


def _serialize_ellipse(shape: Dict[str, Any]) -> List[str]:
    lines = [_tag("ellipse", name=shape.get("nm", ""), ix=shape.get("ix"))]
    lines.extend(_serialize_property_tag("ellipse_position", shape.get("p")))
    lines.extend(_serialize_property_tag("ellipse_size", shape.get("s")))
    lines.append("(/ellipse)")
    return lines


def _serialize_property_tag(tag: str, prop: Dict[str, Any] | None) -> List[str]:
    if not prop:
        return []
    if prop.get("a", 0) == 1:
        lines = [f"({tag} animated=true)"]
        for keyframe in prop.get("k", []):
            lines.append(_tag("keyframe", **_keyframe_attrs(keyframe)))
        lines.append(f"(/{tag})")
        return lines

    value = prop.get("k")
    if isinstance(value, dict) and {"x", "y"} <= set(value.keys()):
        values = [value["x"], value["y"]]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    return [f"({tag} {' '.join(_fmt(v) for v in values if v is not None)})"]


def _keyframe_attrs(keyframe: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {"t": keyframe.get("t", 0)}
    start = keyframe.get("s")
    if isinstance(start, list):
        if start and isinstance(start[0], dict):
            pass
        else:
            attrs["s"] = " ".join(_fmt(v) for v in start)
    elif start is not None:
        attrs["s"] = start
    for prefix in ("i", "o"):
        tangent = keyframe.get(prefix)
        if tangent:
            x_val = tangent.get("x")
            y_val = tangent.get("y")
            if x_val is not None:
                attrs[f"{prefix}_x"] = _space_join(x_val)
            if y_val is not None:
                attrs[f"{prefix}_y"] = _space_join(y_val)
    if keyframe.get("h") is not None:
        attrs["h"] = keyframe["h"]
    if keyframe.get("to") is not None:
        attrs["to"] = json.dumps(keyframe["to"], ensure_ascii=True)
    if keyframe.get("ti") is not None:
        attrs["ti"] = json.dumps(keyframe["ti"], ensure_ascii=True)
    if keyframe.get("e") is not None and not isinstance(keyframe["e"], list):
        attrs["e"] = keyframe["e"]
    return attrs


def _path_closed(prop: Dict[str, Any]) -> bool:
    value = prop.get("k", {})
    if isinstance(value, dict):
        return bool(value.get("c", True))
    return True


def _first_scalar(prop: Dict[str, Any] | None, default: float = 0) -> float:
    if not prop:
        return default
    value = prop.get("k")
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            start = value[0].get("s", [default])
            return float(start[0] if isinstance(start, list) else start)
        return float(value[0])
    return float(value if value is not None else default)


def _first_vector(prop: Dict[str, Any] | None, fallback: List[float]) -> List[float]:
    if not prop:
        return fallback
    value = prop.get("k")
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            start = value[0].get("s", fallback)
            if isinstance(start, list) and start and isinstance(start[0], dict):
                return fallback
            return list(start)
        return list(value)
    return fallback


def _static_property_text(prop: Dict[str, Any] | None) -> str | None:
    if not prop:
        return None
    value = prop.get("k")
    if isinstance(value, list) and value and not isinstance(value[0], dict):
        return " ".join(_fmt(v) for v in value)
    return None


def _tag(tag_name: str, **attrs: Any) -> str:
    parts = [f"({tag_name}"]
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            if key in {"s", "i_x", "i_y", "o_x", "o_y"} and '"' not in value and key != "s":
                rendered = value
            elif key in {"s", "position", "scale", "anchor"} and '"' not in value:
                rendered = value
            else:
                rendered = f'"{_escape(value)}"'
        else:
            rendered = _fmt(value)
        parts.append(f" {key}={rendered}")
    parts.append(")")
    return "".join(parts)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _space_join(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_fmt(v) for v in value)
    return _fmt(value)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

