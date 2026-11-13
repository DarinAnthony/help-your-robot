"""Pinned repository URMA extractor configurations shared by both registries."""

from __future__ import annotations

from molmo_spaces.paths import PI_EMBODIMENT_ASSETS_DIR, REPO
from molmo_spaces.training import sep1_conditioning_contract as sep1_contract
from pi05_integration import input_config


def franka9_urma2_adarms(
    fusion: str = "residual_mlp",
) -> input_config.ExtractedEmbodimentAdaRMS:
    """Nine-embodiment URMA2 artifact, built lazily from the canonical robot bank."""
    return input_config.ExtractedEmbodimentAdaRMS(
        extractor=input_config.Urma2EmbodimentExtractor(
            bundle_dir=f"{PI_EMBODIMENT_ASSETS_DIR}/20260720_urma2_franka9",
            bundle_sha256=(
                "a66e3d195d81b6c486b48b0b8ee753ee5e5fe465f8063c4f4501b056bb5a7753"
            ),
            build_manifest=f"{REPO}/configs/embodiment/20260720_urma2_franka9.json",
            build_manifest_sha256=(
                "eab3e8eb29bfd4ef9923f0e35bc7a9070db5b1433a2891b901b913070d5483d5"
            ),
        ),
        fusion=fusion,
    )


def franka_remaining81_urma2_concat_project_adarms() -> (
    input_config.ExtractedEmbodimentConcatProjectAdaRMS
):
    """Remaining81 URMA2 with 1024-wide morphology and timestep concat projection."""
    return input_config.ExtractedEmbodimentConcatProjectAdaRMS(
        extractor=input_config.Urma2EmbodimentExtractor(
            bundle_dir=f"{PI_EMBODIMENT_ASSETS_DIR}/20260723_urma2_franka_remaining81",
            bundle_sha256=(
                "a1ecb954b106d21319ddb6a2ab0245be846b57f6eb220f6ad85b837059854fc3"
            ),
            build_manifest=(
                f"{REPO}/configs/embodiment/20260723_urma2_franka_remaining81.json"
            ),
            build_manifest_sha256=(
                "8ea5a71af2d00a81ebb8412ca29f5fb1b8dad64dd63d27c82bffaa31ac8d6599"
            ),
        )
    )


def franka_remaining81_urma2_backbone_token() -> (
    input_config.ExtractedEmbodimentBackboneToken
):
    """Remaining81 URMA2 projected into one PaliGemma prefix token."""
    return input_config.ExtractedEmbodimentBackboneToken(
        extractor=input_config.Urma2EmbodimentExtractor(
            bundle_dir=f"{PI_EMBODIMENT_ASSETS_DIR}/20260723_urma2_franka_remaining81",
            bundle_sha256=(
                "a1ecb954b106d21319ddb6a2ab0245be846b57f6eb220f6ad85b837059854fc3"
            ),
            build_manifest=(
                f"{REPO}/configs/embodiment/20260723_urma2_franka_remaining81.json"
            ),
            build_manifest_sha256=(
                "8ea5a71af2d00a81ebb8412ca29f5fb1b8dad64dd63d27c82bffaa31ac8d6599"
            ),
        ),
        token_width=2048,
    )


def franka_oer_hash8095_urma2_concat_project_adarms() -> (
    input_config.ExtractedEmbodimentConcatProjectAdaRMS
):
    """OER catalog descriptions with historical 1024-wide AdaRMS fusion."""
    return input_config.ExtractedEmbodimentConcatProjectAdaRMS(
        extractor=input_config.Urma2EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                "20260728_urma2_oer_quality_hold_50k_catalog_v1"
            ),
            bundle_sha256=(
                "b7a1eebe7574d4118af10dbade6b122a3b755fc3a16c7e22ed170168204a6b6c"
            ),
        )
    )


def franka_oer_hash8095_urma3_concat_project_adarms() -> (
    input_config.ExtractedEmbodimentConcatProjectAdaRMS
):
    """OER URMA3 controls, normalized on the exact 8,095-embodiment view."""
    return input_config.ExtractedEmbodimentConcatProjectAdaRMS(
        extractor=input_config.Urma3EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                "20260728_urma3_oer_hash8095_catalog_v1"
            ),
            bundle_sha256=(
                "45da190f23f445f8657a96004b9876c538a7d4e0e362b1318a3c7be572b0929f"
            ),
        )
    )


def ur5e_trossen_v6_camd8k_urma3_concat_project_adarms(
    *, bundle_sha256: str
) -> input_config.ExtractedEmbodimentConcatProjectAdaRMS:
    """URMA3 conditioning pinned to the exact Camera-D v6 8K robot view."""
    return input_config.ExtractedEmbodimentConcatProjectAdaRMS(
        extractor=input_config.Urma3EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                "20260829_urma3_ur5e_trossen_v6_camd8k_v1"
            ),
            bundle_sha256=bundle_sha256,
        )
    )


def ur5e_trossen_v6_camd8k_urma4_concat_project_adarms(
    *, bundle_sha256: str
) -> input_config.ExtractedEmbodimentConcatProjectAdaRMS:
    """URMA4 joint/global conditioning for the frozen Camera-D v6 8K view."""

    return input_config.ExtractedEmbodimentConcatProjectAdaRMS(
        extractor=input_config.Urma4EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{sep1_contract.URMA4_BUNDLE_NAME}"
            ),
            bundle_sha256=bundle_sha256,
        )
    )


def ur5e_trossen_v6_camd8k_urma4_backbone_token(
    *, bundle_sha256: str
) -> input_config.ExtractedEmbodimentBackboneToken:
    """The same URMA4 joint/global record projected into one VLM prefix token."""

    return input_config.ExtractedEmbodimentBackboneToken(
        extractor=input_config.Urma4EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                f"{sep1_contract.URMA4_BUNDLE_NAME}"
            ),
            bundle_sha256=bundle_sha256,
        ),
        token_width=2048,
    )


def ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_token(
    *, bundle_sha256: str
) -> input_config.ExtractedEmbodimentBackboneToken:
    """URMA4 prefix conditioning for the frozen Camera-D 4K bridge view."""

    return input_config.ExtractedEmbodimentBackboneToken(
        extractor=input_config.Urma4EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                "20260909_urma4_ur5e_discreteaxes_slidegripper_cam2520_4k_v1/source"
            ),
            bundle_sha256=bundle_sha256,
        ),
        token_width=2048,
    )


def ur5e_discreteaxes_slidegripper_cam2520_4k_urma4_backbone_token_v2(
    *, bundle_sha256: str
) -> input_config.ExtractedEmbodimentBackboneToken:
    """URMA4 prefix conditioning for the RLDS-index-exact Camera-D 4K bridge."""

    return input_config.ExtractedEmbodimentBackboneToken(
        extractor=input_config.Urma4EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                "20260909_urma4_ur5e_discreteaxes_slidegripper_cam2520_4k_v2/source"
            ),
            bundle_sha256=bundle_sha256,
        ),
        token_width=2048,
    )


def ur5e_bridge4k_plus_calibrated_anchor_urma4_backbone_token(
    *, bundle_sha256: str
) -> input_config.ExtractedEmbodimentBackboneToken:
    """URMA4 prefix conditioning for bridge-4K plus the calibrated center anchor."""

    return input_config.ExtractedEmbodimentBackboneToken(
        extractor=input_config.Urma4EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                "20260910_urma4_bridge4k_plus_calibrated_center_anchor_v1/source"
            ),
            bundle_sha256=bundle_sha256,
        ),
        token_width=2048,
    )


def ur5e_twoemb800_urma3_concat_project_adarms(
    *, bundle_sha256: str
) -> input_config.ExtractedEmbodimentConcatProjectAdaRMS:
    """URMA3 conditioning for the exact ID355 + random-v6 two-robot study."""

    return input_config.ExtractedEmbodimentConcatProjectAdaRMS(
        extractor=input_config.Urma3EmbodimentExtractor(
            bundle_dir=(
                f"{PI_EMBODIMENT_ASSETS_DIR}/"
                "20260831_urma3_ur5e_twoemb_id0355_i013877_v1"
            ),
            bundle_sha256=bundle_sha256,
        )
    )
