"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro
from etils import epath

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if np.asarray(v).dtype.kind not in ("U", "S", "O")}


class NormalizationInputs(transforms.DataTransformFn):
    """Extract the numeric fields needed by normalization without parsing images.

    ``component_ids`` also keeps the loader's ``dataset_component_id`` (a string) for component-weighted mixture
    statistics."""

    def __init__(self, component_ids: bool = False):
        self.component_ids = component_ids

    def __call__(self, data: dict) -> dict:
        result = {
            "state": data["observation/state"],
            "actions": data["actions"],
        }
        if "embodiment_id" in data:
            result["embodiment_id"] = data["embodiment_id"]
        if self.component_ids:
            result["dataset_component_id"] = data["dataset_component_id"]
        for key in (
            "response_state_history",
            "response_state_history_mask",
            "response_action_history",
            "response_action_history_mask",
            # Holonomic base (mobile_base_slots_v1/v2 only).
            "base_state",
            "base_state_mask",
            "base_actions",
            "base_action_mask",
            # Chunked scans of the query-anchored arm actions only.
            "action_loss_mask",
        ):
            if key in data:
                result[key] = data[key]
        return result


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int | None]:
    # MolmoSpaces normalization is canonical and horizon-independent: update statistics
    # from each real trajectory action exactly once, before future chunk construction. This
    # holds for DROID mixtures too: chunking them counted every action ~horizon times and mixed
    # the synthetic past-the-end chunk targets into the quantiles. A query-anchored arm action
    # exists only as a chunk (its label grows with the offset from the query step), so its
    # statistics pool the chunk positions the loss sees (compute's valid_values).
    if not data_config.input_config.arm_action.query_anchored:
        action_horizon = None
    dataset = _data_loader.create_rlds_dataset(
        data_config,
        action_horizon,
        batch_size,
        shuffle=False,
        deterministic=True,
        num_parallel_reads=1,
        num_parallel_calls=1,
        repeat=False,
        # Normalization only consumes state and actions. Preserve the canonical
        # loader order and transforms without decoding two JPEGs per frame.
        decode_images=False,
        # Prompt overlays are intentionally outside the normalization contract.
        # Statistics consume only unchanged state/action tensors.
        apply_behavior_prompt_overlay=False,
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            # The normal training transform also parses images. JPEG decoding is intentionally
            # disabled for this numeric-only scan, so reproduce its state/action/ID projection
            # directly without touching image fields.
            NormalizationInputs(component_ids=data_config.input_config.normalization.component_weighted),
        ],
        is_batched=True,
    )
    # This is a finite, deterministic traversal. Do not wrap it in RLDSDataLoader,
    # whose training semantics intentionally restart an exhausted iterator.
    del max_frames
    return dataset, None


# Tables up to this many values reduce as one float64 table (16 GiB); larger ones one column at a time.
WHOLE_TABLE_MAX_ELEMENTS = 2**31


def _exact_statistics(batches: list[np.ndarray]) -> normalize.NormStats:
    """Exact per-dimension statistics of every row. Consumes ``batches``."""
    if not batches:
        raise ValueError("normalization group contains no frames")
    flattened = [np.asarray(batch).reshape(-1, np.asarray(batch).shape[-1]) for batch in batches]
    width = flattened[0].shape[1]
    if any(batch.shape[1] != width for batch in flattened):
        raise ValueError("normalization batches have different feature widths")
    rows = sum(len(batch) for batch in flattened)
    if rows < 2:
        raise ValueError("normalization group contains fewer than two frames")
    if rows * width <= WHOLE_TABLE_MAX_ELEMENTS:
        values = np.concatenate(flattened, axis=0)
        batches.clear()
        flattened.clear()
        # The whole float64 table: bitwise the historical result, so recorded assets recompute exactly.
        values = values.astype(np.float64, copy=False)
        return normalize.NormStats(
            mean=np.mean(values, axis=0),
            std=np.std(values, axis=0),
            q01=np.quantile(values, 0.01, axis=0),
            q99=np.quantile(values, 0.99, axis=0),
        )
    # Gather each column directly instead of first concatenating the entire
    # N x D float32 table. The contiguous float64 column, its row order, and all
    # reductions are byte-identical to the historical large-table path.
    columns = []
    for column in range(width):
        x = np.concatenate([batch[:, column] for batch in flattened]).astype(np.float64)
        columns.append((np.mean(x), np.std(x), *np.quantile(x, [0.01, 0.99])))
        del x
    batches.clear()
    flattened.clear()
    mean, std, q01, q99 = (np.array(stat) for stat in zip(*columns))
    return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def _exact_position_statistics(batches: list[np.ndarray]) -> normalize.NormStats:
    """Per-position (H, D) statistics of (N, H, D) chunks whose positions past the episode end are NaN. Consumes
    ``batches`` (cleared once reduced)."""
    if not batches:
        raise ValueError("normalization group contains no frames")
    shape = np.asarray(batches[0]).shape[1:]
    if sum(np.asarray(batch).size for batch in batches) <= WHOLE_TABLE_MAX_ELEMENTS:
        values = np.concatenate(batches, axis=0).astype(np.float64, copy=False)
        batches.clear()
        if np.min(np.sum(~np.isnan(values), axis=0)) < 2:
            raise ValueError("a chunk position has fewer than two in-episode labels")
        return normalize.NormStats(
            mean=np.nanmean(values, axis=0),
            std=np.nanstd(values, axis=0),
            q01=np.nanquantile(values, 0.01, axis=0),
            q99=np.nanquantile(values, 0.99, axis=0),
        )
    # pop34fk at H50 holds a 131 GB float32 payload shared by global/group views, so a
    # concatenated float64 copy does not fit. Each (k, d) cell is gathered from the batches and reduced on its own in
    # float64: the same statistics, up to the summation order of the mean and std.
    stats = {name: np.empty(shape) for name in ("mean", "std", "q01", "q99")}
    for cell in np.ndindex(*shape):
        x = np.concatenate([np.asarray(batch)[(slice(None), *cell)] for batch in batches]).astype(np.float64)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            raise ValueError("a chunk position has fewer than two in-episode labels")
        stats["mean"][cell], stats["std"][cell] = np.mean(x), np.std(x)
        stats["q01"][cell], stats["q99"][cell] = np.quantile(x, [0.01, 0.99])
    batches.clear()
    return normalize.NormStats(**stats)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """Column-wise quantile of (N, M) values under (N, M) sample weights (zero = absent), interpolating the
    weighted midpoint CDF."""
    result = np.empty(values.shape[1])
    for column in range(values.shape[1]):
        present = weights[:, column] > 0
        order = np.argsort(values[present, column], kind="stable")
        value, weight = values[present, column][order], weights[present, column][order]
        result[column] = np.interp(q, (np.cumsum(weight) - 0.5 * weight) / weight.sum(), value)
    return result


def _component_weighted_statistics(
    batches: list[np.ndarray], components: list[np.ndarray], weights: dict[str, float]
) -> normalize.NormStats:
    """Statistics of the mixture that samples component c with probability ``weights[c]``: every component spreads
    its weight evenly over its own valid labels (per position for (N, H, D) chunks, whose past-the-end positions are
    NaN), whatever its frame count."""
    if not batches:
        raise ValueError("normalization group contains no frames")
    values = np.concatenate(batches, axis=0).astype(np.float64, copy=False)
    ids = np.concatenate(components, axis=0)
    if set(ids) != set(weights):
        raise ValueError(f"component-weighted statistics saw components {sorted(set(ids))}, configured {sorted(weights)}")
    valid = ~np.isnan(values)
    sample_weight = np.zeros_like(values)
    for component, weight in weights.items():
        rows = ids == component
        counts = np.sum(valid[rows], axis=0)
        if np.min(counts) < 2:
            raise ValueError(f"component {component} has fewer than two valid labels at some position")
        sample_weight[rows] = np.where(valid[rows], weight / counts, 0.0)
    sample_weight /= np.sum(sample_weight, axis=0)
    filled = np.where(valid, values, 0.0)
    mean = np.sum(sample_weight * filled, axis=0)
    std = np.sqrt(np.sum(sample_weight * np.where(valid, filled - mean, 0.0) ** 2, axis=0))
    flat_values, flat_weights = filled.reshape(len(values), -1), sample_weight.reshape(len(values), -1)
    return normalize.NormStats(
        mean=mean,
        std=std,
        q01=_weighted_quantile(flat_values, flat_weights, 0.01).reshape(mean.shape),
        q99=_weighted_quantile(flat_values, flat_weights, 0.99).reshape(mean.shape),
    )


def _selected_row_views(
    value: np.ndarray, selected: np.ndarray, row_edges: np.ndarray | None = None
) -> list[np.ndarray]:
    """Select rows in their original order without duplicating the chunk payload.

    Each contiguous run is a basic slice sharing the global table. Concatenating
    these views yields exactly ``value[selected]``, including float32 NaN bits.
    Optional ``row_edges`` maps each frame to its flattened in-episode chunk
    rows for pooled action statistics. The reducers receive the same row order.
    """
    indices = np.flatnonzero(selected)
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    chunks = []
    for part in np.split(indices, boundaries):
        start, end = part[0], part[-1] + 1
        if row_edges is not None:
            start, end = row_edges[start], row_edges[end]
        if end > start:
            chunks.append(value[start:end])
    for chunk in chunks:
        chunk.setflags(write=False)
    return chunks


def compute(config: _config.TrainConfig, max_frames: int | None = None):
    """Compute statistics for an already-resolved native training config."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    if data_config.input_config.response_history.enabled:
        keys += ["response_state_history", "response_action_history"]
    # The base gets its own tables over mobile frames only; "state"/"actions" keep their arm-only
    # values, so a fixed-arm mixture's statistics are identical with and without the option.
    base_mask_keys = {"base_state": "base_state_mask", "base_actions": "base_action_mask"}
    # Per-embodiment tables stay arm-only: the base uses one global table (MobileBaseSlotsV1/V2).
    embodiment_keys = list(keys)
    if data_config.input_config.mobile_base.enabled:
        keys += list(base_mask_keys)
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    normalization_groups = ()
    group_for_id: dict[int, str] = {}
    group_values: dict[tuple[str, str], list[np.ndarray]] = {}
    if data_config.embodiment_normalization:
        normalization_ids = data_config.embodiment_normalization_ids
        if not normalization_ids:
            raise ValueError("per-embodiment normalization requires normalization IDs")
        normalization_groups = tuple(
            getattr(data_config.input_config.normalization, "groups", ())
        )
        if normalization_groups:
            for group in normalization_groups:
                for embodiment_id in group.embodiment_ids:
                    group_for_id[int(embodiment_id)] = group.name
                for key in embodiment_keys:
                    group_values[(group.name, key)] = []
            if set(group_for_id) != set(normalization_ids):
                raise ValueError(
                    "grouped normalization membership must cover exactly the configured "
                    "normalization IDs"
                )
        else:
            values.update({
                transforms.embodiment_norm_key(key, embodiment_id): []
                for embodiment_id in normalization_ids
                for key in embodiment_keys
            })
        seen_ids = set()

    frame_count = 0
    chunked_actions = data_config.rlds_data_dir is not None and data_config.input_config.arm_action.query_anchored
    # Opt-in (GroupedChunkPositionStandardNorm): (H, D) action tables, optionally weighted by mixture component.
    scheme = data_config.input_config.normalization
    per_position, component_weighted = scheme.per_chunk_position, scheme.component_weighted
    if per_position and not chunked_actions:
        raise ValueError("per-chunk-position statistics need the query-anchored action chunks of an RLDS config")
    if per_position and data_config.input_config.response_history.enabled:
        raise NotImplementedError("per-chunk-position statistics cover the arm state and actions only")
    # With a mobile base the arm keeps its per-position tables and the base its one global (3,) table per key,
    # read from chunk position 0 below (each frame's own base command once, as in an unchunked scan).
    component_weights = {}
    if component_weighted:
        # The loader's dataset_component_id of every component (molmospaces_rlds_dataset.restructure).
        component_weights = {
            str(dataset.dataset_id or dataset.data_subdir or dataset.name): float(dataset.weight)
            for dataset in data_config.datasets
        }
        if len(component_weights) != len(data_config.datasets):
            raise ValueError("component-weighted statistics need unique dataset component IDs")
    value_components: dict[str, list[np.ndarray]] = {key: [] for key in values}
    group_components: dict[tuple[str, str], list[np.ndarray]] = {key: [] for key in group_values}

    def valid_values(batch, key, count, selected=None):
        value = np.asarray(batch[key])[:count]
        if selected is not None:
            value = value[selected]
        if key == "actions" and chunked_actions:
            # (B, H, D) chunks: every in-episode position counts, past-the-end holds do not.
            mask = np.asarray(batch["action_loss_mask"], dtype=bool)[:count]
            if selected is not None:
                mask = mask[selected]
            if per_position:
                # Keep each row's positions apart; the ones the loss masks become NaN.
                if np.isnan(value).any():
                    raise ValueError("query-anchored action labels must be finite")
                # NaN of the label's own dtype: a float64 copy of a full scan's float32 chunks would not fit.
                return np.where(np.any(mask, axis=-1)[..., None], value, value.dtype.type(np.nan))
            value = value[np.any(mask, axis=-1)]
            if value.size == 0:
                return None
        if key.startswith("response_"):
            mask = np.asarray(batch[f"{key}_mask"])[:count]
            if selected is not None:
                mask = mask[selected]
            value = value[mask]
            if value.size == 0:
                return None
        if key in base_mask_keys:
            mask = np.asarray(batch[base_mask_keys[key]], dtype=bool)[:count]
            if selected is not None:
                mask = mask[selected]
            if np.any(np.any(mask, axis=-1) != np.all(mask, axis=-1)):
                raise ValueError("a base mask marks all three coordinates or none")
            if key == "base_actions" and chunked_actions:
                # A query-anchored scan loads (B, H, 3) base chunks; position 0 is the frame's own base command,
                # so the base table equals the unchunked scan's (past-the-end zero holds never count).
                value = value[:, 0]
            value = value[np.all(mask, axis=-1)]
            if value.size == 0:
                return None
        return value

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing exact stats"):
        batch_size = len(np.asarray(batch["state"]))
        if max_frames is not None:
            batch_size = min(batch_size, max_frames - frame_count)
            if batch_size <= 0:
                break
        components = (
            np.asarray([
                value.decode() if isinstance(value, bytes) else str(value)
                for value in np.asarray(batch["dataset_component_id"]).reshape(-1)[:batch_size]
            ])
            if component_weighted
            else None
        )
        pooled_row_edges = None
        if chunked_actions and not per_position and not component_weighted:
            valid_counts = np.any(np.asarray(batch["action_loss_mask"], dtype=bool)[:batch_size], axis=-1).sum(axis=-1)
            pooled_row_edges = np.concatenate(([0], np.cumsum(valid_counts, dtype=np.int64)))
        batch_values = {}
        for key in keys:
            selected_values = valid_values(batch, key, batch_size)
            batch_values[key] = selected_values
            if selected_values is not None:
                values[key].append(selected_values)
                if component_weighted:
                    value_components[key].append(components)
        if data_config.embodiment_normalization:
            embodiment_ids = np.asarray(batch["embodiment_id"])[:batch_size].reshape(-1)
            for embodiment_id in np.unique(embodiment_ids):
                embodiment_id = int(embodiment_id)
                if embodiment_id not in normalization_ids:
                    raise ValueError(
                        f"dataset embodiment_id {embodiment_id} is outside configured codebook"
                    )
                seen_ids.add(embodiment_id)
                selected = embodiment_ids == embodiment_id
                for key in embodiment_keys:
                    if per_position or (chunked_actions and key in {"state", "actions"} and not component_weighted):
                        # The global table already has the same masked labels.
                        # Group views avoid a second H32/H50-sized copy;
                        # keep the historical per-ID ordering within every batch.
                        if batch_values[key] is None:
                            continue
                        chunks = (
                            _selected_row_views(batch_values[key], selected, pooled_row_edges)
                            if key == "actions" and not per_position else
                            _selected_row_views(batch_values[key], selected)
                        )
                        destination = (
                            group_values[(group_for_id[embodiment_id], key)]
                            if normalization_groups else
                            values[transforms.embodiment_norm_key(key, embodiment_id)]
                        )
                        destination.extend(chunks)
                        if component_weighted:
                            destination_components = (
                                group_components[(group_for_id[embodiment_id], key)]
                                if normalization_groups else
                                value_components[transforms.embodiment_norm_key(key, embodiment_id)]
                            )
                            destination_components.extend(_selected_row_views(components, selected))
                        continue
                    selected_values = valid_values(batch, key, batch_size, selected)
                    if selected_values is None:
                        continue
                    if normalization_groups:
                        group_values[(group_for_id[embodiment_id], key)].append(
                            selected_values
                        )
                        if component_weighted:
                            group_components[(group_for_id[embodiment_id], key)].append(components[selected])
                    else:
                        values[transforms.embodiment_norm_key(key, embodiment_id)].append(
                            selected_values
                        )
                        if component_weighted:
                            value_components[transforms.embodiment_norm_key(key, embodiment_id)].append(
                                components[selected]
                            )
        frame_count += batch_size
        if max_frames is not None and frame_count >= max_frames:
            break

    if data_config.embodiment_normalization:
        missing = set(normalization_ids) - seen_ids
        if missing:
            raise ValueError(f"normalization scan did not cover embodiment IDs {sorted(missing)}")

    for key in base_mask_keys:
        if key in values and not values[key]:
            raise ValueError(
                f"mobile-base normalization found no mobile-base frames for {key!r}; "
                "a mobile-base recipe needs at least one mobile episode of its conversion "
                "(mobile_base_feature_v1 for mobile_base_slots_v1, mobile_base_feature_v2 for v2)"
            )
    def statistics(field, batches, components):
        if component_weighted:
            return _component_weighted_statistics(batches, components, component_weights)
        if per_position and field == "actions":
            return _exact_position_statistics(batches)
        return _exact_statistics(batches)

    field_of = {key: key for key in keys}
    if data_config.embodiment_normalization and not normalization_groups:
        field_of.update({
            transforms.embodiment_norm_key(key, embodiment_id): key
            for embodiment_id in normalization_ids
            for key in embodiment_keys
        })
    # Pop each table as it is reduced so that only one is ever duplicated in memory.
    norm_stats = {
        key: statistics(field_of[key], values.pop(key), value_components[key]) for key in list(values)
    }
    if normalization_groups:
        pooled = {
            (group.name, key): statistics(
                key, group_values.pop((group.name, key)), group_components[(group.name, key)]
            )
            for group in normalization_groups
            for key in embodiment_keys
        }
        for group in normalization_groups:
            for embodiment_id in group.embodiment_ids:
                for key in embodiment_keys:
                    norm_stats[
                        transforms.embodiment_norm_key(key, embodiment_id)
                    ] = pooled[(group.name, key)]

    if data_config.input_config.response_history.enabled:
        norm_stats["response_state_history_gaussian"] = norm_stats.pop(
            "response_state_history"
        )
        norm_stats["response_action_history_gaussian"] = norm_stats.pop(
            "response_action_history"
        )

    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("normalization output requires an asset id")
    output_root = (
        config.assets_dirs
        if config.data.assets.assets_dir is None
        else epath.Path(config.data.assets.assets_dir)
    )
    output_path = output_root / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    return output_path, frame_count


def main(config_name: str, max_frames: int | None = None):
    compute(_config.get_config(config_name), max_frames=max_frames)


if __name__ == "__main__":
    tyro.cli(main)
