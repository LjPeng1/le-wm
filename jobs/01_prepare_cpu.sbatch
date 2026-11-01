#!/usr/bin/env bash
#SBATCH --job-name=pusht-cpu-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPUs, memory, wall time, partition and output are provided by submit_stage1.sh.
set -Eeuo pipefail
source "$(dirname "$0")/lib.sh"
job_start prepare_cpu

PUSHT_STEP="install_environment"
command -v uv >/dev/null || { echo "Install/enable uv in the permitted cluster environment first" >&2; exit 2; }
mkdir -p "$(dirname "$PUSHT_VENV")"
if [[ ! -x "$PUSHT_VENV/bin/python" ]]; then
  uv venv --python 3.10 "$PUSHT_VENV"
fi
activate_env
[[ "$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = 3.10 ]]
uv pip install -r requirements/repro-pusht.txt
uv pip install -r requirements/repro-tests.txt
mkdir -p "$PUSHT_RUN_DIR/installs"
python -m pip freeze > "$PUSHT_RUN_DIR/installs/pip-freeze.txt" || uv pip freeze > "$PUSHT_RUN_DIR/installs/pip-freeze.txt"
python -m pytest tests -q -p no:cacheprovider
python -m scripts.repro_pusht.preflight --cache-root "$STABLEWM_HOME" \
  --output "$PUSHT_RUN_DIR/audits/preflight.json"

PUSHT_STEP="download_official_assets"
mkdir -p "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/hf_pusht" "$STABLEWM_HOME/pusht_release"
hf download quentinll/lewm-pusht \
  --revision 22b330c28c27ead4bfd1888615af1340e3fe9052 \
  --local-dir "$STABLEWM_HOME/hf_pusht"
hf download quentinll/lewm-pusht --repo-type dataset \
  --revision 655cd446b9929369d7d406001da85c15d1457850 \
  --local-dir "$STABLEWM_HOME/pusht_release"
python - <<'PY'
import json,os,pathlib,hashlib
root=pathlib.Path(os.environ["STABLEWM_HOME"])
manifest=json.loads((pathlib.Path(os.environ["PUSHT_REPO"])/"docs/PUSHT_SOURCE_MANIFEST.json").read_text())
model=manifest["official_model"]
data=manifest["official_dataset"]
model_weight=next(x for x in model["files"] if x["rfilename"]=="weights.pt")
data_archive=next(x for x in data["files"] if x["rfilename"]=="pusht_expert_train.h5.zst")
checks=[
(root/"hf_pusht/weights.pt",model_weight["lfs"]["sha256"]),
(root/"hf_pusht/config.json",model["config_sha256"]),
(root/"pusht_release/pusht_expert_train.h5.zst",data_archive["lfs"]["sha256"])
]
for path,expected in checks:
  h=hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
  if h.hexdigest()!=expected:raise ValueError(f"Release hash mismatch: {path}")
  print("VERIFIED",path,path.stat().st_size)
PY

PUSHT_STEP="decompress_and_convert"
H5="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
LANCE="$STABLEWM_HOME/datasets/pusht_expert_train.lance"
if [[ ! -e "$H5" ]]; then
  python - <<'PY'
import os,pathlib,zstandard
root=pathlib.Path(os.environ["STABLEWM_HOME"])
with (root/"pusht_release/pusht_expert_train.h5.zst").open("rb") as src:
  with (root/"datasets/pusht_expert_train.h5").open("xb") as dst:
    zstandard.ZstdDecompressor().copy_stream(src,dst)
PY
fi
if [[ ! -e "$LANCE" ]]; then
  python - <<'PY'
import os,pathlib
from stable_worldmodel.data import convert
root=pathlib.Path(os.environ["STABLEWM_HOME"])/"datasets"
convert(str(root/"pusht_expert_train.h5"),str(root/"pusht_expert_train.lance"),
        source_format="hdf5",dest_format="lance",mode="error",jpeg_quality=95)
PY
fi

PUSHT_STEP="gate1_verify_real_data"
python -m scripts.repro_pusht.verify_data \
  --hdf5 "$H5" --lance "$LANCE" \
  --output-dir "$PUSHT_RUN_DIR/audits/data_check" --execute
python - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ["PUSHT_RUN_DIR"])/"audits/data_check/data_verification.json"
d=json.loads(p.read_text())
assert d["status"]=="passed" and d["passed"]
PY
gate_pass gate1

PUSHT_STEP="gate2_convert_and_compare_model"
python -m scripts.repro_pusht.convert_hf \
  --source "$STABLEWM_HOME/hf_pusht" \
  --output "$PUSHT_RUN_DIR/official_pusht" --execute
H5_SHA="$(sha256sum "$H5" | cut -d ' ' -f1)"
printf '%s\n' "$H5_SHA" > "$PUSHT_RUN_DIR/gates/unpacked_h5.sha256"
python -m scripts.repro_pusht.verify_model \
  --source "$STABLEWM_HOME/hf_pusht" \
  --converted "$PUSHT_RUN_DIR/official_pusht" \
  --hdf5 "$H5" --expected-hdf5-sha256 "$H5_SHA" \
  --output "$PUSHT_RUN_DIR/audits/model_check" --execute
python - <<'PY'
import json,os,pathlib
d=json.loads((pathlib.Path(os.environ["PUSHT_RUN_DIR"])/"audits/model_check/completed.json").read_text())
assert d.get("status")=="passed_function_checks"
assert d.get("official_asset_run")
PY
gate_pass gate2
echo "CPU DATA AND MODEL GATES PASSED. Only the dependent approved GPU evaluation may now run."
