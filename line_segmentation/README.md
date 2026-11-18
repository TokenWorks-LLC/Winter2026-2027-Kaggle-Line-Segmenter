# line_segmentation

Run every module from the repository root with `python -m line_segmentation.<folder>.<module>`; each one prints its
options with `--help`. The repository README gives the recipe in order. This file lists what each module does.

All data paths come from `paths.py`: data lives under `tablet_data/` (or `LINECROP_DATA_ROOT`), and `LINE_WEIGHTS`
names the default model. After moving the data, `python -m line_segmentation.paths --rewrite <old-prefix>/` re-roots
the image paths stored in the tables.

## Data: `data/`

| module | what it does | writes |
|---|---|---|
| `ingest/download_ebl.py` | downloads eBL photographs, sign annotations and transliterations; `--scripts` picks the scripts, `--annotations-dump` reads boxes from the eBL annotation export | `tablet_data/ebl_tablets/<script>/<tablet>/`, `manifest.csv` |
| `ingest/extract_atf.py`, `ingest/match_museum.py` | transliterations and museum-number matches from the CDLI dump (`data/cdli/artifacts_json/`) | tables in `data/cdli/artifacts_json/` |
| `ingest/inventory.py`, `ingest/face_qc.py` | inventory and quality check of the Old Assyrian CDLI face photographs | `tablet_data/fat-cross_processed/` |
| `truth/ebl_ground_truth.py` | line truth from eBL sign boxes: one straight segment per line through its sign centres, grouped by side, object and column | `tablet_data/ebl_tablets/ground_truth/` |
| `truth/full_line_annotations.py` | which tablets have every transliterated line annotated | `ground_truth/full_line_annotations.csv` |
| `truth/line_heights.py` | each line's height target (from the sign boxes), for the height outputs | `faces/line_heights.csv` |
| `truth/oa_sample.py`, `truth/profile_lines.py` | the 100 Old Assyrian faces chosen for hand marking, with pre-labels from a projection profile | `fat-cross_processed/ground_truth/` |
| `truth/labeler/`, `truth/labels_from_db.py` | the Old Assyrian Line Marker page used to mark lines, and the conversion of its saved marks | `fat-cross_processed/ground_truth/labels.csv` |
| `faces/prepare_faces.py` | `ebl`: cuts each tablet face out of the composite photograph and moves its line truth into face coordinates; `oa`: the Old Assyrian face list | `tablet_data/ebl_tablets/faces/` |
| `restore_fixed.py` | puts the shipped split and Old Assyrian labels (`data/fixed/`) where the pipeline reads them | |

## Line model: `linemodel/`

| module | what it does |
|---|---|
| `targets.py` | paints the training targets from line segments: kernel (the line's core), band (the full line), orientation and height; `--preview N` draws them |
| `dataset.py` | `build_index` gathers the training faces (eBL tune split with coverage >= 0.9, Old Assyrian labels, Heidelberg lines) and holds out 10% of tablets for validation by seed; `LineWindows` samples augmented 768 px windows |
| `model.py` | the U-Net with a ResNet-34 encoder and the loss |
| `train.py` | training; writes `model.pt` (best validation recall + precision), `last.pt`, `history.csv`, `config.json`, `training_faces.csv`, `plots/`. `--faces-list` reuses an earlier run's faces |
| `infer.py` | prediction on faces (`--split test|tune|all`) and the read-out from maps to line segments (`read_lines`) |
| `visualize.py`, `render_predictions.py` | training graphs, and predictions drawn on faces |
| `list_training_faces.py` | one table of the faces each run used, with CDLI P-numbers; read by `segment_images.py` to record what a model was trained on |

## Running a model: `segment/`

`segment_images.py` runs a model on a folder or a list of images and writes `lines.json`, overlays and optional line
crops; `preprocess.py` loads images and finds and names the views; `views.py` finds tablet views without
annotations. See [segment/README.md](segment/README.md).

## Evaluation: `eval/`

| module | what it does |
|---|---|
| `evaluate_lines.py` | recall, precision, first-line error and count agreement against the eBL truth, by script and resolution |
| `evaluate_heidelberg.py` | the same against hand-drawn Neo-Assyrian lines (Dencker et al. 2020); also cuts the Heidelberg face crops used in training |
| `evaluate_oa.py` | the same against the Old Assyrian hand labels |
| `extend_split.py` | adds new faces to `split.csv` without moving existing ones; `--hold-out` forces listed tablets into the test half |
| `box_containment.py`, `height_calibration.py` | how much of each sign box a line band contains; used to calibrate the height targets |

## Reports and export: `report/`

| module | what it does |
|---|---|
| `export_line_signs.py` | the line dataset: one image per annotated line with its signs in reading order. `--half-height 0.7 --end-margin 0.5` gives `line_signs_std`; `--model-pred` adds the check against the model |
| `stage_kaggle.py` | upload-ready folders for the dataset and the model on Kaggle |
| `report_figures.py` | the figures of `docs/line_segmentation/line_model.md` (needs the stored predictions of each run) |

## Comparisons: `baselines/`

`kraken_seg.py` (Kraken's segmenter, retrained) and `dencker_linenet.py` (the Dencker et al. 2020 line network,
rebuilt); `docs/line_segmentation/related_work.md` gives the results. Kraken runs in its own environment.

## Shared: `crops/crop_frame.py`

The line frame used by every line image: `crop_matrix` rotates the face to a line's angle, and `box_in_frame` maps a
sign box into it.
