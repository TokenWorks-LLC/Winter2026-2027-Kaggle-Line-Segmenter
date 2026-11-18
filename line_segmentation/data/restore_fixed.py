"""Put the small fixed inputs shipped in data/fixed/ where the pipeline reads them.

  python -m line_segmentation.data.restore_fixed [--force]

data/fixed/ebl_split.csv          -> <DATA_ROOT>/ebl_tablets/split.csv (the line model's tune/test split; made once by the
                                     first iteration's evaluation, which is not in this branch, and only extended since)
data/fixed/oa_labels.csv          -> <DATA_ROOT>/fat-cross_processed/ground_truth/labels.csv (hand-marked Old Assyrian lines)
data/fixed/oa_faces_accepted.csv  -> <DATA_ROOT>/fat-cross_processed/prepared/faces_accepted.csv (the labelled faces' rows)

Existing files are left alone unless --force is given.
"""
import argparse
import os
import shutil

from line_segmentation.paths import EBL, OA

FILES = [('data/fixed/ebl_split.csv', os.path.join(EBL, 'split.csv')),
         ('data/fixed/oa_labels.csv', os.path.join(OA, 'ground_truth', 'labels.csv')),
         ('data/fixed/oa_faces_accepted.csv', os.path.join(OA, 'prepared', 'faces_accepted.csv'))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true'); a = ap.parse_args()
    for src, dst in FILES:
        if os.path.exists(dst) and not a.force:
            print(f'kept      {dst} (exists; --force replaces it)'); continue
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(src, dst); print(f'copied    {src} -> {dst}')


if __name__ == '__main__':
    main()
