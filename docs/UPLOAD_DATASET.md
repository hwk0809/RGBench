# Uploading the RGBench Dataset to Hugging Face

Maintainer-only walkthrough. End users download with
[`scripts/download_data.py`](../scripts/download_data.py); this file is
for the one-time upload of fresh data.

The benchmark reads three slices of the on-disk capture tree —
`calibration/`, `joints/`, and `segment_pcds/` — plus the cloth meshes
referenced from `configs/cloth_params/*.yaml`. Everything else under
`~/DataSets/Piper_Data/Official/` (raw `pcd/`, `rgb/`, intrinsics, raw
RGB, etc.) is intermediate data and is NOT uploaded.

## 1. Install + authenticate

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# paste a write-scoped token from https://huggingface.co/settings/tokens
```

## 2. Create the dataset repo

```bash
huggingface-cli repo create RGBench/RGBench-Cloth-Sim2Real-v1 --type dataset
```

Note the `RGBench/` org prefix — RGBench datasets live under the
[RGBench organization](https://huggingface.co/RGBench), not a personal
namespace.

Or via the web UI: <https://huggingface.co/new-dataset>. Public is the
default and is what the benchmark assumes (`download_data.py` does not
pass a token).

## 3. Stage the filtered data

```bash
python scripts/prepare_hf_dataset.py --dry-run    # preview, prints sizes
python scripts/prepare_hf_dataset.py              # actually stage
```

Defaults:
- Source captures: `~/DataSets/Piper_Data/Official/`
- Source meshes:   `~/DataSets/Style3dCloth/`
- Staging dir:     `/tmp/rgbench_hf_upload/`
- Link mode:       hardlink (falls back to copy across filesystems)

Override with `--source-piper`, `--source-meshes`, `--staging`,
`--link-mode {hardlink,symlink,copy}`. For interactive staging on a
filesystem that's tight on space, `--link-mode symlink` skips the copy
entirely — `huggingface-cli` follows symlinks when reading file bytes.

The script also writes `README.md` (HF dataset card) and `LICENSE`
(CC-BY 4.0) into the staging dir.

## 4. Upload

```bash
huggingface-cli upload-large-folder \
    RGBench/RGBench-Cloth-Sim2Real-v1 \
    /tmp/rgbench_hf_upload \
    --repo-type=dataset
```

`upload-large-folder` is resumable — interrupt and re-run it and it
picks up where it left off. ~6.6 GB at a typical home-broadband upload
of 5-20 MB/s takes 10-30 minutes.

## 5. Verify

```bash
# From a different directory or machine
python scripts/download_data.py --sample-only --target /tmp/_rgbench_check
ls /tmp/_rgbench_check/green_tshirt
```

Should print the smoke-test capture directory.

## Versioning

The repo id is `RGBench/RGBench-Cloth-Sim2Real-v1`. For breaking
changes (new captures, mesh format changes, recalibration) bump the
suffix (`v2`, `v3`, ...) instead of overwriting v1, so existing papers
that cite v1 remain reproducible. Minor additions (new garments, more
samples) can stay on v1 — Hugging Face datasets are git-versioned so
you can pin to a revision with `--revision <commit-sha>`.

## Known gaps

- `Beige_Hoodie/Hoodie_Flat_Simple_25k_adjusted.usda` is referenced by
  `configs/cloth_params/beige_hoodie.yaml` but not present in
  `~/DataSets/Style3dCloth/`. Isaac Sim runs on beige_hoodie will fail
  until this is generated and re-staged.
