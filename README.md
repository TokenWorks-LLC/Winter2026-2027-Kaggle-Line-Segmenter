# Cuneiform text lines: a learned line model and a line dataset

This repository shows how a learned text-line model and annotated tablet photographs are turned into a dataset of
single text lines, each with its cuneiform signs in reading order. It contains:

- **the line model**: a U-Net (ResNet-34 encoder) that finds the text lines on a photograph of a tablet face, with the
  code to build its training data, train it and evaluate it (`line_segmentation/linemodel/`);
- **the pipeline to run a finished model** on any images, which writes each line's polygon, baseline and angle and,
  optionally, one image per line (`line_segmentation/segment/`);
- **`line_signs_std`**, a dataset of 14,115 text lines with 83,952 sign boxes from 1,968 faces of 1,183 tablets in the
  Electronic Babylonian Library (eBL), cut with one geometry for every line and checked line by line against the model.
  A 20-face sample is in [`examples/line_signs_std_sample/`](examples/line_signs_std_sample/); the full set is a Kaggle
  dataset.

The work started as a fork of [ProtoSnap](https://github.com/TAU-VAILab/ProtoSnap) (sign-prototype alignment). This
branch keeps only the line model and the dataset; ProtoSnap and the earlier experiments remain on the `main` branch.

## How the dataset is made

```text
eBL photographs + sign boxes ──► line truth per face ──► line model (trained on it) ──► check of every truth line
             │                                                                                   │
             └──────────────► one line image per truth line, fixed geometry ◄────────────────────┘
                                         line_signs_std
```

1. **Line truth from annotations.** eBL annotators drew a box around each sign and assigned it to a transliteration
   line. `data/truth/ebl_ground_truth.py` fits a straight segment through each line's sign centres, and
   `data/faces/prepare_faces.py` cuts each tablet face out of the composite museum photograph.
2. **The line model** learns to find lines from these faces, plus hand-marked Old Assyrian faces and hand-drawn
   Neo-Assyrian lines (Dencker et al. 2020). It needs no transliteration, so it also works on photographs without
   annotations.
3. **One geometry for every line.** `report/export_line_signs.py` cuts each annotated line out of the face rotated to
   the line's own angle, 0.7 line spacings above and below its middle and 0.5 spacings beyond its first and last sign,
   and lists the signs inside it in reading order with their eBL readings and boxes.
4. **The model checks the annotations.** For every exported line the distance to the nearest line found by the model
   is stored (`model_offset_pitch`); a line with no model line within 0.35 spacings is flagged `model_disagrees`, which
   usually means a misplaced sign box or a damaged line. Other flags mark lines with few signs, no transliteration,
   a transliteration that belongs to another line, odd angles or signs out of order.

On photographs without annotations, step 3 is replaced by the model itself: `segment/segment_images.py --crops` cuts
one image per line the model finds.

## Quick start: run the model on the example images

```bash
pip install -r requirements.txt
pip install huggingface_hub   # provides the hf command
hf download FactGridCuneiform/Winter2026-2027-Kaggle-Line-Segmenter run6_heights_s1/model.pt --local-dir weights/linemodel
python -m line_segmentation.segment.segment_images --images examples/images --out output/example --crops --overlays 5
```

`output/example/lines.json` lists every line found (polygon, bounding box, baseline, angle, score) with the model's
provenance, `overlays/` draws them on the photographs and `crops/` holds one image per line. Use `--views auto` for
photographs that show several sides of a tablet at once; [`segment/README.md`](line_segmentation/segment/README.md)
describes every option and the output format. It runs on a CPU, at a few seconds per image.

## The dataset: `line_signs_std`

Each line has one image, is tied to the face it came from and lists its signs in reading order. The files are
`faces.csv`, `lines.csv` (one row per line, with its transliteration, geometry, quality measures and flags),
`signs.csv` (one row per sign: reading, sign name, box in the face and in the line image) and `lines/<face>/<line_id>.jpg`.
Lines with no flags and a `sign_share` of at least 0.8 are a good default (5,304 lines). The full description, including
the coordinate frames and caveats, is in [`examples/line_signs_std_sample/README.md`](examples/line_signs_std_sample/README.md).

**Licence.** The dataset and the model weights are released under CC BY-NC-SA 4.0. The line images are cut from
photographs shown in the eBL (<https://www.ebl.lmu.de/>). The eBL sign dataset (Lewenstein et al. 2026,
<https://doi.org/10.5334/johd.503>) publishes the photographs of the annotated tablets and crops cut from them under
CC BY-NC 4.0, and the British Museum offers its photographs under CC BY-NC-SA 4.0; CC BY-NC-SA 4.0 satisfies both. Credit
the eBL, the holding museum and this dataset, and do not use the images commercially without the museums' permission.
See [Licences and data sources](#licences-and-data-sources) for the other data.

## The model

| | |
|---|---|
| architecture | U-Net, ResNet-34 encoder (ImageNet weights), input 1,024 px on the long side, 768 px training windows |
| outputs | line kernel, line band, orientation (sin 2θ, cos 2θ) and line height (2 channels), at a quarter of the input resolution |
| read-out | kernels are thresholded, joined to their band, fitted with a segment and extended along the band (`linemodel/infer.py:read_lines`) |
| training data | 507 faces: 396 eBL (Old and Neo-Babylonian, Old and Neo-Assyrian), 84 hand-marked Old Assyrian CDLI faces, 27 Heidelberg faces with hand-drawn lines; 49 validation faces |
| training | 40 epochs, AdamW, one-cycle schedule, mixed precision; the checkpoint with the best validation recall + precision is kept (about an hour on one GPU) |
| eBL test faces (761) | recall 0.94, precision 0.91, first line within half a line spacing on 87% of faces |
| hand-drawn test lines (9 Heidelberg faces) | recall 0.97, precision 1.00 |

Recall is the share of annotated lines matched by a predicted line within 0.35 line spacings; precision is measured only
on faces with at least 90% of their signs annotated. The full report, with the design, every run and the comparison
with Kraken and the Dencker et al. line network, is [`docs/line_segmentation/line_model.md`](docs/line_segmentation/line_model.md).

## Reproducing everything

All commands run from the repository root. Data goes under `tablet_data/` (set `LINECROP_DATA_ROOT` to move it).

**1. Get the data.**

| source | what | how |
|---|---|---|
| eBL | photographs, sign boxes and transliterations | `python -m line_segmentation.data.ingest.download_ebl --scripts NA OB OA NB --annotations-dump <eBL annotation export .json>` (the annotation export is the eBL sign dataset above) |
| Heidelberg (Dencker et al. 2020) | hand-drawn Neo-Assyrian lines | `line_annotations_{train,test}.csv` and `tablet_segments_{train,test}.csv` from <https://github.com/compvis/cuneiform-sign-detection-dataset> into `data/heidelberg/`, and the Heidelberg photographs of the eBL sign-detection data (<https://zenodo.org/records/10693501>) into `data/ebl/ebl_detection_2024/`; `python -m line_segmentation.eval.evaluate_heidelberg --weights weights/linemodel/run6_heights_s1/model.pt` then cuts the face crops into `tablet_data/heidelberg/faces/` |
| CDLI | Old Assyrian face photographs for the 93 hand-marked faces | the obverse/reverse photographs of the P-numbers in `data/fixed/oa_faces_accepted.csv` from <https://cdli.earth/>, saved as `P<number>_<side>.png` (the `file` column) in `tablet_data/fat-cross_processed/fat-cross_processed/`. CDLI's terms allow non-commercial use and do not allow republishing them, so they are not included here |
| this repository | the tune/test split and the Old Assyrian hand labels | `python -m line_segmentation.data.restore_fixed` |

**2. Build the line truth and faces.**

```bash
python -m line_segmentation.data.truth.ebl_ground_truth
python -m line_segmentation.data.faces.prepare_faces ebl
python -m line_segmentation.eval.extend_split            # adds faces that are not in the shipped split; never moves existing ones
python -m line_segmentation.data.truth.line_heights      # per-line height targets
python -m line_segmentation.data.truth.full_line_annotations
python -m line_segmentation.linemodel.targets --preview 12   # optional: draw the training targets
```

**3. Train.** `--faces-list` trains on exactly the faces and roles run 6 used; without it, the faces come from the
seeded split of whatever data is present.

```bash
MPLBACKEND=Agg python -m line_segmentation.linemodel.train --epochs 40 --seed 1 --heidelberg --heights \
    --faces-list weights/linemodel/run6_heights_s1/training_faces.csv --plot-every 5 --out weights/linemodel/my_run
```

The run folder gets `model.pt` (best checkpoint), `last.pt`, `history.csv`, `config.json`, `training_faces.csv` and
`plots/`. Add `--dry-run` first to check the setup in a few seconds.

**4. Evaluate.**

```bash
python -m line_segmentation.linemodel.infer --weights weights/linemodel/my_run/model.pt --faces tablet_data/ebl_tablets/faces/faces.csv --out tablet_data/ebl_tablets/linemodel/compare/my_run --split test
python -m line_segmentation.eval.evaluate_lines --faces tablet_data/ebl_tablets/faces/faces.csv --pred tablet_data/ebl_tablets/linemodel/compare/my_run
python -m line_segmentation.eval.evaluate_heidelberg --weights weights/linemodel/my_run/model.pt
```

**5. Export the dataset.** Run the model on every face, then cut the lines. `--scripts NA NB OA OB` reproduces the
published set; leave it out to include every script downloaded.

```bash
python -m line_segmentation.linemodel.infer --faces tablet_data/ebl_tablets/faces/faces.csv --out tablet_data/ebl_tablets/linemodel/compare/run6_heights_s1_allfaces --split all
python -m line_segmentation.report.export_line_signs --scripts NA NB OA OB --half-height 0.7 --end-margin 0.5 \
    --model-pred tablet_data/ebl_tablets/linemodel/compare/run6_heights_s1_allfaces --out tablet_data/exports/line_signs_std
python -m line_segmentation.report.stage_kaggle --owner <kaggle user>   # upload-ready folders for the two Kaggle datasets
```

## Repository layout

| path | contents |
|---|---|
| `line_segmentation/linemodel/` | model, training targets, data loading, training, inference and read-out |
| `line_segmentation/segment/` | running a finished model on any images |
| `line_segmentation/data/` | downloads (`ingest/`), line truth (`truth/`), face cutting (`faces/`), `restore_fixed.py` |
| `line_segmentation/eval/` | line scoring, Heidelberg scoring, the split, height calibration |
| `line_segmentation/report/` | the dataset export, Kaggle staging and the report figures |
| `line_segmentation/baselines/` | Kraken and the Dencker et al. line network, for comparison |
| `weights/linemodel/run6_heights_s1/` | the default model's config, training history, face list and plots (`model.pt` is on [Hugging Face](https://huggingface.co/FactGridCuneiform/Winter2026-2027-Kaggle-Line-Segmenter)) |
| `data/fixed/` | the split and the Old Assyrian hand labels |
| `examples/` | example photographs and a sample of `line_signs_std` |
| `docs/line_segmentation/` | the model report, related work and data notes |

[`line_segmentation/README.md`](line_segmentation/README.md) describes every module.

## Licences and data sources

| data | terms | in this repository or on Kaggle? |
|---|---|---|
| eBL photographs and sign annotations | CC BY-NC 4.0 as published in the eBL sign dataset; British Museum photographs CC BY-NC-SA 4.0 | yes: `line_signs_std`, the sample and the example images, under CC BY-NC-SA 4.0 |
| CDLI photographs (Old Assyrian training faces) | CDLI terms of use: for non-commercial use, owned by the holding institutions, no publication without permission; no Creative Commons licence | no: only the P-numbers and the hand-marked line positions; download the photographs from CDLI |
| Heidelberg line annotations (Dencker et al. 2020) | the GitHub repository states no licence; its photographs come from CDLI and, for seven tablets, the Vorderasiatisches Museum Berlin, for research use | no: download them from the original repositories |
| model weights | trained on the data above | yes, on Hugging Face, under CC BY-NC-SA 4.0 |
| code | no licence file yet | this repository |

This summary is not legal advice; check each source's terms before reuse.

## Citation and sources

- The line model and dataset: this repository.
- eBL sign annotations: Lewenstein et al. 2026, *Journal of Open Humanities Data*, <https://doi.org/10.5334/johd.503>.
- Hand-drawn lines: Dencker, Klinkisch, Maul and Ommer 2020, *Deep learning of cuneiform sign detection with weak
  supervision using transliteration alignment*, PLOS ONE.
- ProtoSnap: Mikulinsky et al., ICLR 2025, <https://arxiv.org/abs/2502.00129>.
