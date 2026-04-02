from __future__ import annotations

from pathlib import PurePosixPath
from typing import Dict, Optional


DEFAULT_TROP2_CONFIG = (
    "configs/sam2.1_training/sam2.1_hiera_b+_trop2_structural_priors.yaml"
)
DEFAULT_ABLATION_ROOT = PurePosixPath("checkpoints/ablations")
DEFAULT_INFER_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


ABLATION_PRESETS: Dict[str, Dict[str, object]] = {
    "baseline": {
        "with_contain": False,
        "with_contrast": False,
        "loss_contain": 0.0,
        "loss_struct_contrast": 0.0,
        "return_obj_ptr_for_loss": False,
    },
    "contain": {
        "with_contain": True,
        "with_contrast": False,
        "loss_contain": 1.0,
        "loss_struct_contrast": 0.0,
        "return_obj_ptr_for_loss": False,
    },
    "contrast": {
        "with_contain": False,
        "with_contrast": True,
        "loss_contain": 0.0,
        "loss_struct_contrast": 0.2,
        "return_obj_ptr_for_loss": True,
    },
    "full": {
        "with_contain": True,
        "with_contrast": True,
        "loss_contain": 1.0,
        "loss_struct_contrast": 0.2,
        "return_obj_ptr_for_loss": True,
    },
}


def infer_ablation_name(
    ablation: Optional[str] = None,
    with_contain: bool = False,
    with_contrast: bool = False,
) -> Optional[str]:
    if ablation is not None:
        if ablation not in ABLATION_PRESETS:
            raise ValueError(f"Unsupported ablation: {ablation}")
        return ablation
    if with_contain and with_contrast:
        return "full"
    if with_contain:
        return "contain"
    if with_contrast:
        return "contrast"
    return None


def get_ablation_preset(
    ablation: Optional[str] = None,
    with_contain: bool = False,
    with_contrast: bool = False,
) -> Optional[Dict[str, object]]:
    resolved_name = infer_ablation_name(
        ablation=ablation,
        with_contain=with_contain,
        with_contrast=with_contrast,
    )
    if resolved_name is None:
        return None
    return {"name": resolved_name, **ABLATION_PRESETS[resolved_name]}


def default_experiment_dir(ablation_name: str) -> str:
    return str(DEFAULT_ABLATION_ROOT / ablation_name)


def default_checkpoint_path(ablation_name: str) -> str:
    return str(DEFAULT_ABLATION_ROOT / ablation_name / "checkpoints" / "checkpoint.pt")


def build_training_overrides(
    *,
    ablation: Optional[str] = None,
    with_contain: bool = False,
    with_contrast: bool = False,
    experiment_dir: Optional[str] = None,
) -> tuple[list[str], Optional[str]]:
    preset = get_ablation_preset(
        ablation=ablation,
        with_contain=with_contain,
        with_contrast=with_contrast,
    )
    if preset is None:
        return [], None

    experiment_dir = experiment_dir or default_experiment_dir(preset["name"])
    overrides = [
        f"++trainer.loss.all.weight_dict.loss_contain={preset['loss_contain']}",
        (
            "++trainer.loss.all.weight_dict.loss_struct_contrast="
            f"{preset['loss_struct_contrast']}"
        ),
        (
            "++trainer.model.return_obj_ptr_for_loss="
            f"{str(bool(preset['return_obj_ptr_for_loss'])).lower()}"
        ),
        f"++launcher.experiment_log_dir={experiment_dir}",
    ]
    return overrides, str(preset["name"])
