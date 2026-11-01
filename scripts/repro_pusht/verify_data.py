"""Opt-in audit of the released HDF5 and its official SWM Lance conversion.

The row/index/numeric audit is exhaustive. JPEG decoding and reader-window
checks are bounded deterministic samples, never a claim of pixel equivalence.
No data is downloaded or converted by this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .common import require_execute, sha256, write_json


REQUIRED_COLUMNS = ("pixels", "action", "proprio", "state")
INDEX_COLUMNS = ("episode_idx", "step_idx")
FRAMESKIP, HISTORY_SIZE, NUM_PREDS, JPEG_QUALITY = 5, 3, 1, 95


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_entries(roots):
    entries = []
    for label in ("hdf5", "lance"):
        root = Path(roots[label])
        if root.is_symlink():
            raise ValueError(f"Data root must not be a symlink: {root}")
        if label == "hdf5":
            if not root.is_file():
                raise FileNotFoundError(root)
            files = [root]
        else:
            if not root.is_dir():
                raise NotADirectoryError(root)
            children = sorted(root.rglob("*"))
            if any(p.is_symlink() for p in children):
                raise ValueError("Lance artifacts must not contain symlinks")
            files = [p for p in children if p.is_file()]
            if not files:
                raise ValueError("Lance artifact directory is empty")
        for path in files:
            before = path.stat()
            checksum = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Artifact changed while hashing: {path}")
            entries.append({"root": label,
                            "path": "." if label == "hdf5" else path.relative_to(root).as_posix(),
                            "bytes": after.st_size, "sha256": checksum})
    return entries


def content_manifest(hdf5_path, lance_path):
    """Hash every byte in the HDF5 file and every file in the Lance table."""
    roots = {"hdf5": str(Path(hdf5_path).resolve()),
             "lance": str(Path(lance_path).resolve())}
    manifest = {"schema_version": 1, "kind": "pusht_data_content", "roots": roots,
                "artifacts": _artifact_entries(roots)}
    return {**manifest, "digest": _digest(manifest)}


def validate_content_manifest(path):
    """Rehash full artifacts, reject missing/extra/changed files; return identity.

    This is intentionally expensive and is used before accepting a training
    identity. It checks bytes, not scientific/data-verification acceptance.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "kind", "roots", "artifacts", "digest"}:
        raise ValueError("Unexpected data content manifest fields")
    unsigned = {k: v for k, v in manifest.items() if k != "digest"}
    if manifest["schema_version"] != 1 or manifest["kind"] != "pusht_data_content":
        raise ValueError("Unsupported data content manifest")
    if manifest["digest"] != _digest(unsigned):
        raise ValueError("Data content manifest digest mismatch")
    roots = manifest["roots"]
    if set(roots) != {"hdf5", "lance"} or any(not Path(p).is_absolute() for p in roots.values()):
        raise ValueError("Data manifest requires absolute HDF5/Lance roots")
    for entry in manifest["artifacts"]:
        if set(entry) != {"root", "path", "bytes", "sha256"}:
            raise ValueError("Invalid data artifact entry")
        relative = PurePosixPath(entry["path"])
        if entry["root"] not in roots or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid data artifact path")
    current = content_manifest(roots["hdf5"], roots["lance"])
    if current != manifest:
        raise ValueError("Data artifacts changed, disappeared, or were added")
    return manifest


def _official_api():
    # Lazy import: --execute is required before optional/large dependencies.
    from stable_worldmodel.data import HDF5Dataset, LanceDataset
    from stable_worldmodel.data.formats.lance import _encode_frame
    return HDF5Dataset, LanceDataset, _encode_frame


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sample_indices(total, count):
    if total <= 0:
        return []
    count = min(total, count)
    # Integer arithmetic includes both boundaries with no RNG/version drift.
    return [0] if count == 1 else [i * (total - 1) // (count - 1) for i in range(count)]


def _integer_vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return value.astype(np.int64)


def _metadata(source, target):
    lengths = _integer_vector(source.lengths, "HDF5 ep_len")
    offsets = _integer_vector(source.offsets, "HDF5 ep_offset")
    other_lengths = _integer_vector(target.lengths, "Lance lengths")
    other_offsets = _integer_vector(target.offsets, "Lance offsets")
    if not len(lengths) or len(offsets) != len(lengths) or np.any(lengths <= 0):
        raise ValueError("Empty/invalid HDF5 episode structure")
    expected_offsets = np.cumsum(np.r_[0, lengths[:-1]])
    if not np.array_equal(lengths, other_lengths) or not np.array_equal(expected_offsets, other_offsets):
        raise ValueError("Lance episode lengths/offsets differ from the official conversion")
    total = int(lengths.sum())
    intervals = sorted((int(o), int(o + n)) for o, n in zip(offsets, lengths))
    cursor = 0
    for start, end in intervals:
        if start != cursor or end <= start:
            raise ValueError("HDF5 episode row ranges overlap or leave unaudited gaps")
        cursor = end
    if cursor != total:
        raise ValueError("HDF5 episode row coverage mismatch")
    return lengths, offsets, other_offsets, total


def _source_schema(source, total):
    # The pinned reader's native handle exposes raw shape/dtype without reading
    # image payloads. All actual values/windows still use the official readers.
    with source._open_h5() as file:
        schema = {name: {"shape": list(file[name].shape), "dtype": str(file[name].dtype)}
                  for name in source.column_names}
    for name, value in schema.items():
        if not value["shape"] or value["shape"][0] != total:
            raise ValueError(f"HDF5 column {name} does not cover every episode row")
    return schema


def _index_encoding(values):
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind not in "iu" or not len(values):
        raise ValueError("Source index columns must contain integers")
    vals = [int(x) for x in values]
    if len(vals) == 1 or all(x == vals[0] for x in vals):
        return {"kind": "constant", "value": vals[0], "count": len(vals)}
    step = vals[1] - vals[0]
    if all(x == vals[0] + i * step for i, x in enumerate(vals)):
        return {"kind": "arithmetic", "start": vals[0], "step": step, "count": len(vals)}
    return {"kind": "values", "values": vals}


def _compare_numeric(actual, original, *, rows, boundary, label):
    actual, original = _array(actual), _array(original)
    expected = original.reshape(rows, -1)
    if expected.dtype.kind not in "biuf" or actual.dtype != np.float32:
        raise ValueError(f"{label}: expected numeric source and official float32 target")
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: numeric shape mismatch {expected.shape} != {actual.shape}")
    if np.isinf(expected).any() or np.isinf(actual).any():
        raise ValueError(f"{label}: unexpected infinity")
    with np.errstate(over="ignore", invalid="ignore"):
        cast = expected.astype(np.float32)
    if np.isinf(cast).any():
        raise ValueError(f"{label}: float32 conversion overflows")
    nan = np.isnan(cast)
    if not np.array_equal(nan, np.isnan(actual)):
        raise ValueError(f"{label}: unmatched NaN")
    if np.any(nan & ~np.asarray(boundary, dtype=bool)[:, None]):
        raise ValueError(f"{label}: NaN outside an episode boundary")
    if not np.array_equal(cast, actual, equal_nan=True):
        raise ValueError(f"{label}: values differ after the official float32 cast")
    finite = np.isfinite(expected)
    error = np.abs(expected[finite].astype(np.float64) - cast[finite].astype(np.float64))
    return {"rows": rows, "elements": int(actual.size), "matching_boundary_nan_elements": int(nan.sum()),
            "max_float32_rounding_abs": float(error.max()) if error.size else 0.0}


def _loss_metrics(original, decoded):
    difference = decoded.astype(np.float64) - original.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    return {"elements": int(difference.size), "mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(mse)), "max_abs": float(np.max(np.abs(difference))),
            "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
            "pixel_identical": bool(not np.any(difference))}


def _pixel_check(source, target, ep, step, column, encode_frame):
    original = _array(source._load_slice(ep, step, step + 1)[column])
    decoded = _array(target._load_slice(ep, step, step + 1)[column])
    if original.ndim != 4 or original.shape[0] != 1 or original.shape[1] != 3 or original.dtype != np.uint8:
        raise ValueError(f"{column}: expected uint8 RGB source frames")
    if decoded.shape != original.shape or decoded.dtype != np.uint8:
        raise ValueError(f"{column}: decoded JPEG shape/dtype mismatch")
    blob = encode_frame(original[0].transpose(1, 2, 0), JPEG_QUALITY)
    expected = _array(target._decode_images([blob]))
    if not np.array_equal(decoded, expected):
        raise ValueError(f"{column}: sampled decoded frame differs from official JPEG95 replay")
    return {"episode": int(ep), "step": int(step), "column": column,
            "official_jpeg95_replay_decoded_exact": True, **_loss_metrics(original, decoded)}


def _close(*readers):
    for reader in readers:
        handle = getattr(reader, "h5_file", None)
        if handle is not None:
            handle.close()
            reader.h5_file = None


def verify_data(hdf5_path, lance_path, output_dir, *, execute=False,
                pixel_samples=16, window_samples=16, batch_size=4096):
    """Verify existing data with pinned SWM readers and write exclusive evidence.

    Acceptance is exact after the writer's documented float32 cast. NaNs are
    accepted only when both backends match at first/last episode rows. JPEG
    samples must match a replay of the official quality-95 encoder/decoder;
    original-vs-decoded loss is measured, with no arbitrary loss threshold.
    All clip indices are checked; only the reported windows are fetched.
    """
    if not execute:
        raise ValueError("Data verification requires --execute; no data was read")
    for name, value in (("pixel_samples", pixel_samples), ("window_samples", window_samples), ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    hdf5, lance, output = (Path(p).resolve() for p in (hdf5_path, lance_path, output_dir))
    if output == hdf5 or output.is_relative_to(lance) or hdf5.is_relative_to(lance):
        raise ValueError("Audit output and HDF5 must be outside the Lance artifact root")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "passed": False,
              "protocol": {"frameskip": FRAMESKIP, "history_size": HISTORY_SIZE,
                           "num_preds": NUM_PREDS, "jpeg_quality": JPEG_QUALITY},
              "scope": {"artifact_hashes": "full", "row_index_mapping": "full",
                        "numeric_values": "full", "decoded_pixels": "deterministic_sample",
                        "reader_windows": "deterministic_sample",
                        "full_image_decode": False,
                        "pixel_claim": "lossy JPEG conversion, not original-pixel equivalence"}}
    readers = []
    try:
        manifest = content_manifest(hdf5, lance)
        HDF5Dataset, LanceDataset, encode_frame = _official_api()
        def open_reader(cls, path, **kwargs):
            # Metadata/row readers do not need millions of single-frame clips.
            # The official base class materializes clip_indices eagerly.
            kwargs.setdefault("num_steps", 2 ** 31)
            reader = cls(path=path.as_posix(), **kwargs)
            readers.append(reader)
            return reader
        source = open_reader(HDF5Dataset, hdf5)
        target = open_reader(LanceDataset, lance)
        report["readers"] = {"hdf5": type(source).__module__ + "." + type(source).__qualname__,
                             "lance": type(target).__module__ + "." + type(target).__qualname__}
        lengths, offsets, target_offsets, total = _metadata(source, target)
        source_schema = _source_schema(source, total)
        if "step_idx" not in source.column_names or not {"episode_idx", "ep_idx"}.intersection(source.column_names):
            raise ValueError("HDF5 is missing the source episode/step indices used by the original evaluation")
        names = {key: key.replace(".", "_") for key in source.column_names if key not in INDEX_COLUMNS}
        if len(set(names.values())) != len(names) or set(names.values()) != set(target.column_names):
            raise ValueError("Source/target column mapping differs from the official writer schema")
        if not set(REQUIRED_COLUMNS) <= names.keys():
            raise ValueError("Missing required Push-T columns")
        image_names = [name for name, target_name in names.items() if target_name in target.image_columns]
        if image_names != ["pixels"]:
            raise ValueError("Expected the original Push-T pixels image column only")
        numeric = [name for name in names if name not in image_names]
        source_numeric = open_reader(HDF5Dataset, hdf5, keys_to_load=numeric)
        target_numeric = open_reader(LanceDataset, lance, keys_to_load=[names[k] for k in numeric])
        source_indices = {name: _integer_vector(source.get_col_data(name), f"source {name}")
                          for name in (*INDEX_COLUMNS, "ep_idx") if name in source.column_names}
        target_indices = {name: _integer_vector(target.get_col_data(name), f"target {name}")
                          for name in INDEX_COLUMNS}
        if any(len(values) != total for values in target_indices.values()):
            raise ValueError("Target index column length mismatch")
        stats = {name: {"rows": 0, "elements": 0, "matching_boundary_nan_elements": 0,
                        "max_float32_rounding_abs": 0.0} for name in numeric}
        mapping_path = output / "row_mapping.jsonl"
        with mapping_path.open("x", encoding="utf-8") as mapping:
            for ep, (length, source_offset, target_offset) in enumerate(zip(lengths, offsets, target_offsets)):
                length, source_offset, target_offset = int(length), int(source_offset), int(target_offset)
                target_slice = slice(target_offset, target_offset + length)
                if not np.all(target_indices["episode_idx"][target_slice] == ep):
                    raise ValueError(f"Episode {ep}: target episode_idx was not reindexed by official writer")
                if not np.array_equal(target_indices["step_idx"][target_slice], np.arange(length)):
                    raise ValueError(f"Episode {ep}: target step_idx mismatch")
                record = {"episode": ep, "length": length,
                          "hdf5_row_start": source_offset, "lance_row_start": target_offset,
                          "mapping_rule": "hdf5_row_start+i -> lance_row_start+i, 0<=i<length",
                          "target_episode_idx": ep, "target_step_idx": {"start": 0, "step": 1, "count": length},
                          "source_index_columns": {name: _index_encoding(values[source_offset:source_offset + length])
                                                   for name, values in source_indices.items()}}
                mapping.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                for start in range(0, length, batch_size):
                    stop = min(start + batch_size, length)
                    src = source_numeric.get_row_data(list(range(source_offset + start, source_offset + stop)))
                    dst = target_numeric.get_row_data(list(range(target_offset + start, target_offset + stop)))
                    boundary = np.isin(np.arange(start, stop), [0, length - 1])
                    for name in numeric:
                        result = _compare_numeric(dst[names[name]], src[name], rows=stop - start,
                                                  boundary=boundary, label=f"episode {ep}/{name}/rows {start}:{stop}")
                        for key in ("rows", "elements", "matching_boundary_nan_elements"):
                            stats[name][key] += result[key]
                        stats[name]["max_float32_rounding_abs"] = max(stats[name]["max_float32_rounding_abs"], result["max_float32_rounding_abs"])
        report["full_audit"] = {"episodes": len(lengths), "rows": total, "index_rows": total,
                                "numeric": stats, "source_schema": source_schema,
                                "target_schema": str(target._connect_table().schema),
                                "column_mapping": names,
                                "source_index_policy": "source IDs preserved in mapping; target IDs regenerated by episode order",
                                "numeric_acceptance": "exact after official float32 cast; matched NaN allowed only at first/last episode rows",
                                "row_mapping": {"path": mapping_path.name, "sha256": sha256(mapping_path),
                                                "episodes": len(lengths), "rows": total}}
        pixel_source = open_reader(HDF5Dataset, hdf5, keys_to_load=image_names)
        pixel_target = open_reader(LanceDataset, lance, keys_to_load=[names[k] for k in image_names])
        pixel_results = []
        for logical_row in _sample_indices(total, pixel_samples):
            ep = int(np.searchsorted(target_offsets, logical_row, side="right") - 1)
            step = int(logical_row - target_offsets[ep])
            pixel_results.append(_pixel_check(pixel_source, pixel_target, ep, step, "pixels", encode_frame))
        report["pixel_samples"] = {"requested": pixel_samples, "checked_frames": len(pixel_results),
                                   "population_frames": total, "samples": pixel_results,
                                   "acceptance": "exact decoded replay of pinned official JPEG95 codec; loss metrics are descriptive"}
        steps, span = HISTORY_SIZE + NUM_PREDS, (HISTORY_SIZE + NUM_PREDS) * FRAMESKIP
        window_source = open_reader(HDF5Dataset, hdf5, keys_to_load=list(REQUIRED_COLUMNS), frameskip=FRAMESKIP, num_steps=steps)
        window_target = open_reader(LanceDataset, lance, keys_to_load=list(REQUIRED_COLUMNS), frameskip=FRAMESKIP, num_steps=steps)
        clip_count = sum(max(0, int(length) - span + 1) for length in lengths)
        expected_clips = ((ep, start) for ep, length in enumerate(lengths)
                          for start in range(max(0, int(length) - span + 1)))
        if (len(window_source.clip_indices) != clip_count or len(window_target.clip_indices) != clip_count
                or any(left != expected or right != expected
                       for left, right, expected in zip(window_source.clip_indices, window_target.clip_indices, expected_clips))):
            raise ValueError("Actual reader clip indices differ from the frameskip5/history3+pred1 protocol")
        if not clip_count:
            raise ValueError("No complete frameskip5/history3+pred1 windows")
        windows = []
        for idx in _sample_indices(clip_count, window_samples):
            ep, start = window_source.clip_indices[idx]
            left, right = window_source[idx], window_target[idx]
            raw = source_numeric.get_row_data(list(range(int(offsets[ep]) + start, int(offsets[ep]) + start + span)))
            observation_steps = list(range(start, start + span, FRAMESKIP))
            for name in REQUIRED_COLUMNS[1:]:
                original = _array(raw[name]) if name == "action" else _array(raw[name])[::FRAMESKIP]
                expected = original.reshape(steps, -1)
                source_actual = _array(left[name]).reshape(steps, -1)
                if not np.array_equal(source_actual, expected, equal_nan=True):
                    raise ValueError(f"Window {idx}/{name}: HDF5 reader alignment mismatch")
                boundary = [pos in (0, int(lengths[ep]) - 1) for pos in observation_steps]
                # Action rows are grouped; a boundary NaN can occur only in the
                # first/last action group, and full row audit already enforced
                # its precise raw-row location.
                if name == "action":
                    boundary = [start + group * FRAMESKIP == 0 or start + (group + 1) * FRAMESKIP == int(lengths[ep])
                                for group in range(steps)]
                _compare_numeric(_array(right[name]), expected, rows=steps, boundary=boundary,
                                 label=f"window {idx}/{name}")
            frame_records = []
            for j, step in enumerate(observation_steps):
                result = _pixel_check(pixel_source, pixel_target, ep, step, "pixels", encode_frame)
                expected_left = _array(pixel_source._load_slice(ep, step, step + 1)["pixels"])[0]
                expected_right = _array(pixel_target._load_slice(ep, step, step + 1)["pixels"])[0]
                if not np.array_equal(_array(left["pixels"])[j], expected_left) or not np.array_equal(_array(right["pixels"])[j], expected_right):
                    raise ValueError(f"Window {idx}: decoded observation alignment mismatch")
                frame_records.append(result)
            windows.append({"clip_index": idx, "episode": ep, "start": start,
                            "observation_steps": observation_steps,
                            "action_steps": {"start": start, "stop_exclusive": start + span,
                                             "reshape": [steps, int(_array(right["action"]).shape[-1])]},
                            "decoded_frames": frame_records})
        report["window_samples"] = {"requested": window_samples, "population_windows": clip_count,
                                    "full_clip_indices_checked": clip_count, "checked_windows": len(windows),
                                    "checked_frame_pairs": len(windows) * steps, "samples": windows}
        # Detect mutation during the audit; a manifest never binds pre-read
        # bytes to later values without another complete byte check.
        if content_manifest(hdf5, lance) != manifest:
            raise ValueError("Data artifacts changed during verification")
        write_json(output / "data_content_manifest.json", manifest)
        report.update(status="passed", passed=True, mismatch_count=0,
                      data_content_manifest_digest=manifest["digest"])
    except Exception as error:
        report.update(status="failed", passed=False, mismatch_count=None,
                      error={"type": type(error).__name__, "message": str(error)})
        write_json(output / "data_verification.json", report)
        raise
    finally:
        _close(*readers)
    write_json(output / "data_verification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-samples", type=int, default=16)
    parser.add_argument("--window-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require_execute(parser, args)
    report = verify_data(args.hdf5, args.lance, args.output_dir, execute=True,
                         pixel_samples=args.pixel_samples, window_samples=args.window_samples,
                         batch_size=args.batch_size)
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "data_verification.json")}))


if __name__ == "__main__":
    main()
