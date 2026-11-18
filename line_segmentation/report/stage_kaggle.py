"""Stage the two Kaggle datasets in upload-ready folders (nothing is uploaded).

  python -m line_segmentation.report.stage_kaggle --owner <kaggle username> [--out tablet_data/kaggle] [--copy]

Writes
  <out>/line-signs-std/      the full tablet_data/exports/line_signs_std (README, CSVs, lines/) + dataset-metadata.json
  <out>/cuneiform-line-model/ weights/linemodel/run6_heights_s1/{model.pt, config.json, history.csv, training_faces.csv}
                             + a README + dataset-metadata.json
Files are hard-linked when the file system allows it (no extra space), else copied; --copy always copies.
Upload each folder with the Kaggle CLI, e.g. `kaggle datasets create -p tablet_data/kaggle/line-signs-std --dir-mode zip`.
Check the licence fields before uploading: the images are eBL/museum photographs (British Museum CC BY-NC-SA 4.0; others vary).
"""
import argparse
import json
import os
import shutil

from line_segmentation.paths import LINE_WEIGHTS

MODEL_README = """# Cuneiform text-line model (run 6)

A U-Net with a ResNet-34 encoder that finds text lines on photographs of cuneiform tablet faces. It predicts a line
kernel, a line band, the line orientation and each line's height; `line_segmentation.linemodel.infer.read_lines`
turns those maps into one segment per line.

Put `model.pt` at `weights/linemodel/run6_heights_s1/model.pt` in the code repository, then run
`python -m line_segmentation.segment.segment_images --images <folder> --out <out> --crops`.

Trained for 40 epochs, seed 1, on 507 faces (eBL photographs of Old Babylonian, Old Assyrian, Neo-Babylonian and
Neo-Assyrian tablets, 84 hand-labelled Old Assyrian CDLI faces and 27 Heidelberg faces with hand-drawn lines) and
validated on 49; `training_faces.csv` lists them. On the eBL test faces it finds 94% of the annotated lines with 91%
precision. The method and results are in `docs/line_segmentation/line_model.md` of the repository.
"""


def place(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst): return
    if not copy:
        try: os.link(src, dst); return
        except OSError: pass
    shutil.copyfile(src, dst)


def stage_tree(src, dst, copy):
    n = 0
    for dp, _, files in os.walk(src):
        for fn in files:
            place(os.path.join(dp, fn), os.path.join(dst, os.path.relpath(dp, src), fn), copy); n += 1
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--owner', required=True, help='Kaggle username or organisation that will own the datasets')
    ap.add_argument('--out', default='tablet_data/kaggle'); ap.add_argument('--data', default='tablet_data/exports/line_signs_std')
    ap.add_argument('--copy', action='store_true'); a = ap.parse_args()
    d = os.path.join(a.out, 'line-signs-std'); n = stage_tree(a.data, d, a.copy)
    json.dump({'title': 'Cuneiform text lines with signs (eBL, standardised)', 'id': f'{a.owner}/line-signs-std',
               'licenses': [{'name': 'CC-BY-NC-SA-4.0'}]}, open(os.path.join(d, 'dataset-metadata.json'), 'w'), indent=1)
    print(f'{n} files -> {d}')
    run = os.path.dirname(LINE_WEIGHTS); m = os.path.join(a.out, 'cuneiform-line-model')
    for fn in ('model.pt', 'config.json', 'history.csv', 'training_faces.csv'):
        place(os.path.join(run, fn), os.path.join(m, fn), a.copy)
    open(os.path.join(m, 'README.md'), 'w', encoding='utf-8').write(MODEL_README)
    json.dump({'title': 'Cuneiform text-line model (U-Net, run 6)', 'id': f'{a.owner}/cuneiform-line-model',
               'licenses': [{'name': 'CC-BY-NC-SA-4.0'}]}, open(os.path.join(m, 'dataset-metadata.json'), 'w'), indent=1)
    print(f'model -> {m}')


if __name__ == '__main__':
    main()
