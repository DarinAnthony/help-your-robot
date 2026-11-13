"""Host-side grouped metrics and input/output visualization for OpenPI training.

This mirrors the useful W&B structure of the Octo trainer while keeping plotting and string
formatting outside JIT-compiled model code.
"""

from __future__ import annotations

import base64
import collections
import html
import io

import numpy as np
from PIL import Image

TASK_NAMES = {0: "open", 1: "close", -1: "na"}


def group_names(task_ids, embodiment_ids) -> list[str]:
    """Return stable task/embodiment labels for one flattened batch/window."""
    tasks = np.asarray(task_ids).reshape(-1)
    embodiments = np.asarray(embodiment_ids).reshape(-1)
    return [
        f"{TASK_NAMES.get(int(task), f'task_{int(task)}')}/embodiment_{int(embodiment)}"
        for task, embodiment in zip(tasks, embodiments, strict=True)
    ]


def grouped_means_by_group(values, groups: list[str]) -> dict[str, float]:
    """Mean a per-sample metric by an arbitrary precomputed group label, plus the aggregate."""
    values = np.asarray(values).reshape(-1)
    sums: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    for value, group in zip(values, groups, strict=True):
        sums[group] += float(value)
        counts[group] += 1
    result = {group: sums[group] / counts[group] for group in sums}
    if values.size:
        result["_aggregate"] = float(np.mean(values))
    return result


def grouped_means(values, task_ids, embodiment_ids) -> dict[str, float]:
    """Mean a per-sample metric by task/embodiment, plus the sample-weighted aggregate."""
    return grouped_means_by_group(values, group_names(task_ids, embodiment_ids))


def source_group_names(embodiment_ids) -> list[str]:
    """Real DROID co-training source vs sim, keyed off DROID's -2 metrics-only sentinel set in
    molmospaces_rlds_dataset.restructure_droid. Every other embodiment_id is a MolmoSpaces RLDS
    upload (realrig9k, MolmoBot, ...), so it is bucketed as "sim". This used to be labelled
    "realrig9k_sim", which misnamed MolmoBot-heavy mixtures."""
    embodiments = np.asarray(embodiment_ids).reshape(-1)
    return ["droid_real" if int(value) == -2 else "sim" for value in embodiments]


def source_grouped_means(values, embodiment_ids) -> dict[str, float]:
    """Mean a per-sample metric by real (DROID) vs sim source, plus the aggregate."""
    return grouped_means_by_group(values, source_group_names(embodiment_ids))


def _uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        # Model observations are normalized to [-1, 1].
        image = np.clip((image + 1.0) * 127.5, 0, 255)
    return image.astype(np.uint8)


def _image_uri(image: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(_uint8_image(image)).save(output, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _values(values: np.ndarray, *, limit: int | None = None) -> str:
    array = np.asarray(values).reshape(-1)
    if limit is not None:
        array = array[:limit]
    return " ".join(f"{float(value):.3f}" for value in array)


def _polyline(values: np.ndarray, *, width: int, height: int, low: float, high: float) -> str:
    values = np.asarray(values, np.float64)
    xs = np.asarray([0.0]) if len(values) == 1 else np.linspace(0, width, len(values))
    scale = max(high - low, 1e-8)
    ys = height - (values - low) / scale * height
    return " ".join(f"{x:.1f},{np.clip(y, 0, height):.1f}" for x, y in zip(xs, ys, strict=True))


def _action_svg(target: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> str:
    """Render a dependency-free interactive-friendly SVG action comparison."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    valid = np.asarray(valid, dtype=bool)
    valid_dimensions = np.flatnonzero(np.any(valid, axis=0))
    width, row_height = 720, 66
    rows = []
    for row, dimension in enumerate(valid_dimensions):
        mask = valid[:, dimension]
        shown_target = target[:, dimension]
        shown_prediction = prediction[:, dimension]
        combined = np.concatenate([shown_target[mask], shown_prediction[mask]])
        low = float(np.min(combined)) if combined.size else -1.0
        high = float(np.max(combined)) if combined.size else 1.0
        margin = max((high - low) * 0.08, 1e-3)
        low, high = low - margin, high + margin
        y = row * row_height
        gt = _polyline(shown_target, width=width, height=row_height - 18, low=low, high=high)
        pred = _polyline(shown_prediction, width=width, height=row_height - 18, low=low, high=high)
        mse = float(np.mean(np.square(shown_target[mask] - shown_prediction[mask]))) if np.any(mask) else 0.0
        rows.append(
            f'<g transform="translate(0,{y})">'
            f'<text x="0" y="12" font-size="11">a{dimension} MSE {mse:.5f}</text>'
            f'<line x1="0" y1="{row_height - 18}" x2="{width}" y2="{row_height - 18}" stroke="#ddd"/>'
            f'<polyline points="{gt}" fill="none" stroke="#111" stroke-width="1.5"/>'
            f'<polyline points="{pred}" fill="none" stroke="#e45756" stroke-width="1.5"/>'
            "</g>"
        )
    height = max(row_height, row_height * len(valid_dimensions))
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-height:620px">'
        + "".join(rows)
        + "</svg>"
    )


def data_visualization_html(
    observation,
    actions,
    predictions,
    *,
    count: int,
    physical_state_dim: int,
    state_conditioning_dim: int,
    prompt_conditioning: str,
    state_encoding: str,
    normalization: str,
    action_expert_conditioning: str,
    token_describer=None,
) -> str:
    """Build one self-contained W&B HTML panel from actual train-mode samples.

    Images must already have passed ``preprocess_observation(..., train=True)``.  The caller also
    samples predictions from that same observation, so every visual/model value in this panel has
    one unambiguous provenance.
    """
    images = {key: np.asarray(value) for key, value in observation.images.items()}
    state = np.asarray(observation.state)
    actions = np.asarray(actions)
    predictions = np.asarray(predictions)
    action_mask = (
        np.ones_like(actions, dtype=bool)
        if observation.action_loss_mask is None
        else np.asarray(observation.action_loss_mask, dtype=bool)
    )
    embodiment_ids = (
        np.full(len(state), -1) if observation.embodiment_id is None else np.asarray(observation.embodiment_id)
    )
    task_ids = np.full(len(state), -1) if observation.task_id is None else np.asarray(observation.task_id)
    token_ids = None if observation.tokenized_prompt is None else np.asarray(observation.tokenized_prompt)
    token_mask = (
        None if observation.tokenized_prompt_mask is None else np.asarray(observation.tokenized_prompt_mask)
    )
    robotoken_values = getattr(observation, "robotoken_values", None)
    robotoken_types = getattr(observation, "robotoken_types", None)
    robotoken_references = getattr(observation, "robotoken_reference_ids", None)
    robotoken_mask = getattr(observation, "robotoken_mask", None)
    if robotoken_values is not None:
        robotoken_values = np.asarray(robotoken_values)
        robotoken_types = np.asarray(robotoken_types)
        robotoken_references = np.asarray(robotoken_references)
        robotoken_mask = np.asarray(robotoken_mask, dtype=bool)
    cards = []
    for index in range(min(int(count), len(state))):
        task = TASK_NAMES.get(int(task_ids[index]), f"task_{int(task_ids[index])}")
        physical_end = min(physical_state_dim, state.shape[-1])
        conditioning_end = min(physical_end + state_conditioning_dim, state.shape[-1])
        if token_ids is None:
            decoded, pieces, ids = "not tokenized", (), ()
        else:
            valid_ids = token_ids[index] if token_mask is None else token_ids[index][token_mask[index]]
            ids = tuple(int(value) for value in valid_ids)
            if token_describer is None:
                decoded, pieces = "decoder unavailable", ()
            else:
                decoded, pieces = token_describer(valid_ids)
        cameras = "".join(
            f'<figure><img src="{_image_uri(value[index])}"/><figcaption>{html.escape(key)}</figcaption></figure>'
            for key, value in images.items()
        )
        if robotoken_values is None:
            robotoken_panel = ""
        else:
            valid_robotokens = np.flatnonzero(robotoken_mask[index])
            rows = "\n".join(
                f"{position:03d} type={int(robotoken_types[index, position])} "
                f"refs={robotoken_references[index, position].tolist()} "
                f"values={np.array2string(robotoken_values[index, position], precision=3, suppress_small=True)}"
                for position in valid_robotokens
            )
            robotoken_panel = (
                '<details open><summary>Static RoboToken prefix sequence</summary>'
                f'<p><b>Valid tokens:</b> {len(valid_robotokens)}/{robotoken_values.shape[1]}</p>'
                f'<pre>{html.escape(rows)}</pre></details>'
            )
        cards.append(
            '<section class="sample">'
            f'<h2>Training sample {index}: {html.escape(task)}, embodiment {int(embodiment_ids[index])}</h2>'
            f'<div class="cameras">{cameras}</div>'
            '<div class="routes">'
            f'<span><b>Prompt</b> {html.escape(prompt_conditioning)}</span>'
            f'<span><b>State</b> {html.escape(state_encoding)}</span>'
            f'<span><b>Norm</b> {html.escape(normalization)}</span>'
            f'<span><b>Action expert</b> {html.escape(action_expert_conditioning)}</span>'
            '</div>'
            '<details open><summary>Actual model prefix</summary>'
            f'<pre>{html.escape(decoded)}</pre>'
            f'<p><b>Token IDs ({len(ids)}):</b> {html.escape(" ".join(map(str, ids)))}</p>'
            f'<p><b>Token pieces:</b> {html.escape(" ".join(pieces))}</p>'
            f'<p><b>Token mask:</b> {len(ids)} valid / '
            f'{0 if token_ids is None else token_ids.shape[-1]} total positions</p>'
            '</details>'
            f'{robotoken_panel}'
            '<details open><summary>Normalized state partitions</summary>'
            f'<p><b>physical [0:{physical_end}]:</b> {_values(state[index, :physical_end])}</p>'
            f'<p><b>embodiment [{physical_end}:{conditioning_end}]:</b> '
            f'{_values(state[index, physical_end:conditioning_end]) or "none"}</p>'
            f'<p><b>padding [{conditioning_end}:{state.shape[-1]}]:</b> '
            f'{_values(state[index, conditioning_end:]) or "none"}</p>'
            '</details>'
            '<details open><summary>Physical-unit action chunk: target black, sampled prediction red</summary>'
            f'{_action_svg(actions[index], predictions[index], action_mask[index])}'
            f'<p><b>Valid targets:</b> {int(action_mask[index].sum())}/{action_mask[index].size}; '
            f'<b>horizon:</b> {actions.shape[1]}; <b>model dimensions:</b> {actions.shape[2]}</p>'
            '</details>'
            '</section>'
        )
    return """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:Inter,Arial,sans-serif;color:#1f2937;margin:0;padding:12px;background:#f7f7f8}
.sample{background:white;border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:14px}
h2{font-size:16px;margin:0 0 10px}.cameras{display:flex;gap:8px;overflow-x:auto}
figure{margin:0;min-width:210px}img{width:100%;max-height:210px;object-fit:contain;background:#111}
figcaption{font-size:11px;color:#555}.routes{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
.routes span{background:#eef2ff;border-radius:12px;padding:5px 9px;font-size:11px}
details{border-top:1px solid #eee;padding:8px 0}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;background:#f6f8fa;padding:8px;border-radius:6px}p{font-size:11px;word-break:break-word}
</style></head><body>""" + "".join(cards) + "</body></html>"
