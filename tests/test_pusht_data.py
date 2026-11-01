"""Data-audit logic fixtures; only the optional integration uses real SWM I/O.

The small in-memory readers below emulate the inspected pinned SWM interfaces.
Their passing tests are not evidence that real HDF5/Lance assets were checked.
"""

import io
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.repro_pusht import verify_data as audit


def encode_frame(frame, quality):
    buffer = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_images(blobs):
    frames = []
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            frames.append(np.asarray(image.convert("RGB")).transpose(2, 0, 1))
    return np.stack(frames)


@pytest.fixture
def data_case(tmp_path, monkeypatch):
    lengths = np.array([25, 26], dtype=np.int64)
    offsets = np.array([26, 0], dtype=np.int64)  # Conversion reorders physical rows.
    total = int(lengths.sum())
    pixels = ((np.arange(total * 8 * 9 * 3).reshape(total, 8, 9, 3) * 37) % 256).astype(np.uint8)
    source = {"pixels": pixels,
              "action": np.arange(total * 2, dtype=np.float64).reshape(total, 2) / 7,
              "proprio": np.arange(total * 2, dtype=np.float64).reshape(total, 2) / 11,
              "state": np.arange(total * 5, dtype=np.float64).reshape(total, 5) / 13,
              "episode_idx": np.r_[np.full(26, 12), np.full(25, 99)],
              "step_idx": np.r_[np.arange(100, 126), np.arange(900, 925)]}
    source["action"][26, 0] = np.nan  # Matching episode-start NaN is explicit.
    source["state"][25, -1] = np.nan  # Matching episode-end NaN.
    order = np.r_[np.arange(26, 51), np.arange(26)]
    target = {name: values[order].astype(np.float32).reshape(total, -1)
              for name, values in source.items() if name not in ("pixels", *audit.INDEX_COLUMNS)}
    target["pixels"] = np.array([encode_frame(frame, 95) for frame in pixels[order]], dtype=object)
    target["episode_idx"] = np.repeat(np.arange(2, dtype=np.int64), lengths)
    target["step_idx"] = np.r_[np.arange(25), np.arange(26)]
    case = SimpleNamespace(source=source, target=target, lengths=lengths, offsets=offsets,
                           target_offsets=np.array([0, 25]), numeric_reads=[], window_bad=False,
                           clip_bad=False, schema_extra_rows=False)
    case.hdf5 = tmp_path / "data.h5"
    case.hdf5.write_bytes(b"fixture bytes, not a real HDF5")
    case.lance = tmp_path / "data.lance"
    case.lance.mkdir()
    (case.lance / "part.lance").write_bytes(b"fixture bytes, not a real Lance table")

    class Reader:
        is_lance = False

        def __init__(self, *, path, keys_to_load=None, frameskip=1, num_steps=1):
            self.frameskip, self.num_steps = frameskip, num_steps
            self.span = frameskip * num_steps
            self.lengths = case.lengths.copy()
            self.offsets = (case.target_offsets if self.is_lance else case.offsets).copy()
            self.values = case.target if self.is_lance else case.source
            self.column_names = list(keys_to_load or self.values)
            if self.is_lance:
                self.column_names = [k for k in self.column_names if k not in audit.INDEX_COLUMNS]
            self.image_columns = {"pixels"} if self.is_lance else set()
            self.clip_indices = [(ep, start) for ep, length in enumerate(self.lengths)
                                 for start in range(max(0, int(length) - self.span + 1))]
            if self.is_lance and self.num_steps == 4 and case.clip_bad:
                self.clip_indices[-1] = (1, 0)

        def _open_h5(self):
            values = dict(self.values)
            if case.schema_extra_rows:
                values["state"] = np.r_[values["state"], values["state"][:1]]
            return nullcontext(values)

        def _connect_table(self):
            return SimpleNamespace(schema="fixture: image binary; numeric fixed-list float32; int64 indices")

        def get_col_data(self, column):
            return self.values[column]

        def get_row_data(self, rows):
            case.numeric_reads.append((self.is_lance, tuple(rows)))
            return {name: self.values[name][rows] for name in self.column_names}

        def _decode_images(self, blobs):
            return decode_images(blobs)

        def _load_slice(self, ep, start, end):
            begin = int(self.offsets[ep]) + start
            result = {}
            for name in self.column_names:
                values = self.values[name][begin:begin + end - start]
                if name != "action":
                    values = values[::self.frameskip]
                if name == "pixels":
                    values = decode_images(values) if self.is_lance else values.transpose(0, 3, 1, 2)
                result[name] = values
            return result

        def __getitem__(self, index):
            ep, start = self.clip_indices[index]
            result = self._load_slice(ep, start, start + self.span)
            if "action" in result:
                result["action"] = result["action"].reshape(self.num_steps, -1)
            if self.is_lance and case.window_bad:
                result["proprio"] = result["proprio"][::-1].copy()
            return result

    class HDF5Reader(Reader):
        pass

    class LanceReader(Reader):
        is_lance = True

    monkeypatch.setattr(audit, "_official_api", lambda: (HDF5Reader, LanceReader, encode_frame))
    case.run = lambda **kwargs: audit.verify_data(case.hdf5, case.lance, tmp_path / "audit",
                                                   execute=True, pixel_samples=3, window_samples=3,
                                                   batch_size=7, **kwargs)
    case.output = tmp_path / "audit"
    return case


def test_full_numeric_mapping_float32_and_sampled_loss(data_case):
    report = data_case.run()
    assert report["status"] == "passed" and report["mismatch_count"] == 0
    assert report["full_audit"]["rows"] == 51
    assert report["full_audit"]["numeric"]["action"]["rows"] == 51
    assert report["full_audit"]["numeric"]["action"]["matching_boundary_nan_elements"] == 1
    assert report["full_audit"]["numeric"]["state"]["matching_boundary_nan_elements"] == 1
    assert report["full_audit"]["numeric"]["action"]["max_float32_rounding_abs"] > 0
    assert report["pixel_samples"]["checked_frames"] == 3
    assert report["pixel_samples"]["population_frames"] == 51
    assert any(x["mae"] > 0 for x in report["pixel_samples"]["samples"])
    assert report["scope"]["full_image_decode"] is False
    assert report["window_samples"]["population_windows"] == 13
    assert report["window_samples"]["full_clip_indices_checked"] == 13
    assert report["window_samples"]["checked_windows"] == 3
    assert report["window_samples"]["checked_frame_pairs"] == 12
    assert report["window_samples"]["samples"][0]["observation_steps"] == [0, 5, 10, 15]
    assert report["window_samples"]["samples"][0]["action_steps"]["reshape"] == [4, 10]
    mapping = [json.loads(line) for line in (data_case.output / "row_mapping.jsonl").read_text().splitlines()]
    assert mapping[0]["hdf5_row_start"] == 26 and mapping[0]["lance_row_start"] == 0
    assert mapping[0]["source_index_columns"]["episode_idx"]["value"] == 99
    assert mapping[0]["source_index_columns"]["step_idx"]["start"] == 900
    manifest = audit.validate_content_manifest(data_case.output / "data_content_manifest.json")
    assert report["data_content_manifest_digest"] == manifest["digest"]
    assert manifest == audit.content_manifest(data_case.hdf5, data_case.lance)
    for backend in (False, True):
        audited = {row for is_lance, rows in data_case.numeric_reads if is_lance == backend for row in rows}
        assert audited == set(range(51))


@pytest.mark.parametrize("corruption,message", [
    ("numeric", "values differ"), ("shape", "shape mismatch"),
    ("infinity", "unexpected infinity"), ("interior_nan", "NaN outside"),
    ("unmatched_nan", "unmatched NaN"), ("step", "step_idx mismatch"),
    ("episode", "episode_idx was not reindexed"), ("pixels", "cannot identify image"),
    ("jpeg_quality", "JPEG95 replay"), ("window", "values differ"),
    ("clips", "clip indices differ"), ("metadata", "episode lengths/offsets"),
    ("gap", "overlap or leave"), ("raw_rows", "does not cover"),
    ("missing_indices", "missing the source episode/step"),
])
def test_corrupt_data_fails_with_exclusive_failure_record(data_case, corruption, message):
    case = data_case
    if corruption == "numeric":
        case.target["action"][17, 1] += 1
    elif corruption == "shape":
        case.target["state"] = case.target["state"][:, :-1]
    elif corruption == "infinity":
        case.target["state"][4, 1] = np.inf
    elif corruption == "interior_nan":
        case.source["state"][27, 1] = np.nan
        case.target["state"][1, 1] = np.nan
    elif corruption == "unmatched_nan":
        case.target["action"][0, 0] = 0
    elif corruption == "step":
        case.target["step_idx"][9] = 10
    elif corruption == "episode":
        case.target["episode_idx"][0] = 7
    elif corruption == "pixels":
        case.target["pixels"][0] = b"corrupt JPEG"
    elif corruption == "jpeg_quality":
        case.target["pixels"][0] = encode_frame(case.source["pixels"][26], 50)
    elif corruption == "window":
        case.window_bad = True
    elif corruption == "clips":
        case.clip_bad = True
    elif corruption == "metadata":
        case.target_offsets[1] += 1
    elif corruption == "gap":
        case.offsets[0] -= 1
    elif corruption == "raw_rows":
        case.schema_extra_rows = True
    elif corruption == "missing_indices":
        del case.source["episode_idx"]
    with pytest.raises((ValueError, OSError), match=message):
        case.run()
    record = json.loads((case.output / "data_verification.json").read_text())
    assert record["status"] == "failed" and record["passed"] is False
    assert not (case.output / "data_content_manifest.json").exists()


def test_matching_last_action_nan_and_non_affine_source_indices(data_case):
    # The final complete window includes all 20 raw action rows. A NaN at
    # the actual final row remains valid after grouping, without allowing
    # interior NaNs. Non-affine source indices are recorded losslessly.
    data_case.source["action"][25, 1] = np.nan
    data_case.target["action"][-1, 1] = np.nan
    data_case.source["step_idx"][4] += 7
    report = data_case.run()
    assert report["full_audit"]["numeric"]["action"]["matching_boundary_nan_elements"] == 2
    mapping = [json.loads(line) for line in (data_case.output / "row_mapping.jsonl").read_text().splitlines()]
    assert mapping[1]["source_index_columns"]["step_idx"]["kind"] == "values"
    assert mapping[1]["source_index_columns"]["step_idx"]["values"][4] == 111


def test_execution_and_no_overwrite_guards(data_case, monkeypatch):
    monkeypatch.setattr(audit, "_official_api", lambda: pytest.fail("must not import/read"))
    with pytest.raises(ValueError, match="requires --execute"):
        audit.verify_data(data_case.hdf5, data_case.lance, data_case.output)
    assert not data_case.output.exists()
    with pytest.raises(SystemExit) as error:
        audit.main(["--hdf5", str(data_case.hdf5), "--lance", str(data_case.lance),
                    "--output-dir", str(data_case.output)])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="outside"):
        audit.verify_data(data_case.hdf5, data_case.lance, data_case.lance / "audit", execute=True)
    data_case.output.mkdir()
    with pytest.raises(FileExistsError):
        data_case.run()


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing", "manifest_digest"])
def test_manifest_rechecks_all_artifacts(data_case, mutation):
    data_case.run()
    path = data_case.output / "data_content_manifest.json"
    if mutation == "changed":
        data_case.hdf5.write_bytes(b"different bytes")
    elif mutation == "extra":
        (data_case.lance / "unlisted").write_bytes(b"new")
    elif mutation == "missing":
        (data_case.lance / "part.lance").unlink()
    else:
        manifest = json.loads(path.read_text())
        manifest["digest"] = "0" * 64
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit.validate_content_manifest(path)


def test_official_converter_and_real_readers_when_installed(tmp_path):
    """A real tiny conversion, with no asset download or dependency install."""
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    pytest.importorskip("lancedb")
    swm = pytest.importorskip("stable_worldmodel.data")
    hdf5, lance = tmp_path / "tiny.h5", tmp_path / "tiny.lance"
    lengths = [23, 24]
    rows = sum(lengths)
    with h5py.File(hdf5, "w", libver="latest") as file:
        file["ep_len"], file["ep_offset"] = lengths, [0, 23]
        file["pixels"] = ((np.arange(rows * 8 * 9 * 3).reshape(rows, 8, 9, 3) * 37) % 256).astype(np.uint8)
        for name, dim in (("action", 2), ("proprio", 2), ("state", 5)):
            file[name] = np.arange(rows * dim, dtype=np.float64).reshape(rows, dim) / 7
        file["episode_idx"] = np.repeat([17, 31], lengths)
        file["step_idx"] = np.r_[np.arange(23), np.arange(24)]
    swm.convert(hdf5.as_posix(), lance.as_posix(), source_format="hdf5", dest_format="lance",
                mode="error", jpeg_quality=95, progress=False)
    report = audit.verify_data(hdf5, lance, tmp_path / "actual_audit", execute=True,
                               pixel_samples=3, window_samples=3, batch_size=9)
    assert report["status"] == "passed"
    assert report["full_audit"]["rows"] == rows
