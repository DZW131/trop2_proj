from __future__ import annotations

MEMBRANE_LABEL = "membrane"
NUCLEUS_LABEL = "nucleus"

CHINESE_MEMBRANE_LABEL = "\u80bf\u7624\u7ec6\u80de\u819c"
CHINESE_NUCLEUS_LABEL = "\u80bf\u7624\u7ec6\u80de\u6838"
CHINESE_NUCLEUS_CENTER_LABEL = "\u80bf\u7624\u7ec6\u80de\u6838\u4e2d\u5fc3"

_REGION_LABEL_ALIASES = {
    MEMBRANE_LABEL: (
        MEMBRANE_LABEL,
        "tumor_cell_membrane",
        CHINESE_MEMBRANE_LABEL,
    ),
    NUCLEUS_LABEL: (
        NUCLEUS_LABEL,
        "tumor_cell_nucleus",
        CHINESE_NUCLEUS_LABEL,
    ),
}

_PROMPT_LABEL_ALIASES = {
    MEMBRANE_LABEL: (
        *_REGION_LABEL_ALIASES[MEMBRANE_LABEL],
        "membrane_center",
        "tumor_cell_membrane_center",
    ),
    NUCLEUS_LABEL: (
        *_REGION_LABEL_ALIASES[NUCLEUS_LABEL],
        "nucleus_center",
        "tumor_cell_nucleus_center",
        CHINESE_NUCLEUS_CENTER_LABEL,
    ),
}


def _normalize_label_token(value) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _build_alias_map(alias_table):
    alias_map = {}
    for canonical_label, aliases in alias_table.items():
        for alias in aliases:
            alias_map[_normalize_label_token(alias)] = canonical_label
    return alias_map


_REGION_ALIAS_MAP = _build_alias_map(_REGION_LABEL_ALIASES)
_PROMPT_ALIAS_MAP = _build_alias_map(_PROMPT_LABEL_ALIASES)


def canonicalize_label(label, allow_prompt_aliases=True):
    if label is None:
        return None
    alias_map = _PROMPT_ALIAS_MAP if allow_prompt_aliases else _REGION_ALIAS_MAP
    return alias_map.get(_normalize_label_token(label))


def normalize_label_name(label, allow_prompt_aliases=True):
    canonical_label = canonicalize_label(label, allow_prompt_aliases)
    if canonical_label is not None:
        return canonical_label
    return str(label).strip()


def labels_match(lhs, rhs, allow_prompt_aliases=True):
    lhs_canonical = canonicalize_label(lhs, allow_prompt_aliases)
    rhs_canonical = canonicalize_label(rhs, allow_prompt_aliases)
    if lhs_canonical is not None or rhs_canonical is not None:
        return lhs_canonical is not None and lhs_canonical == rhs_canonical
    return _normalize_label_token(lhs) == _normalize_label_token(rhs)


def collect_shape_points_by_label(shapes, label, allow_prompt_aliases=True):
    target_label = canonicalize_label(label, allow_prompt_aliases)
    collected_points = []
    for item in shapes:
        item_label = item.get("label")
        if target_label is not None:
            if canonicalize_label(item_label, allow_prompt_aliases) != target_label:
                continue
        elif not labels_match(item_label, label, allow_prompt_aliases):
            continue

        points = item.get("points")
        if points is not None:
            collected_points.append(points)
    return collected_points
