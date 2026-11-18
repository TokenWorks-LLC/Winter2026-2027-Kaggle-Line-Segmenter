# line_signs_std, sample

This folder is a 20-face sample of the full `line_signs_std` set, five faces per script, with the same files and columns.
The full set (1,968 faces, 14,115 lines) is published on Kaggle; see the repository README. The description below is
the full set's README, unchanged, so its counts refer to the full set.


This set holds 14115 text lines with 83952 sign boxes, from 1968 faces of 1183 eBL tablets. It was exported on 2026-11-05 with
`python -m line_segmentation.report.export_line_signs`.

Each line has one image, is tied to the face it came from, and lists its signs in reading order. The lines are eBL's: eBL's annotators
assigned the sign boxes to transliteration lines, so these lines are annotations, not model output.

Every line has the same geometry. Its image runs 0.7 line spacings above and below the middle of its signs, which is the median row of their box centres. It runs from the line's first sign to its last plus 0.5 spacings at each end, along the line's own angle. Boxes taller than the image reach past its edge, so their `crop_*` coordinates can be negative or exceed the image size.

## Files

- `faces.csv` has one row per face. Its columns are `face`, `tablet`, `script`, `side`, `object`, `image` (the face photograph, relative to the repository),
  `width`, `height`, `skew_deg` (the face's usual line angle in degrees), `pitch_px` (the line spacing), `coverage`, `n_columns`, `split`, `tablet_level` and `n_lines`.
- `lines.csv` has one row per line. Its columns are `line_id`, `face`, `tablet`, `script`, `side`, `object`, `column`, `line_index`, `line_label`, `atf_text` (the
  transliteration of that line, blank if it was not found), `image`, `x0`, `y0`, `x1` and `y1` (the rectangle in the line frame), `skew_deg`, `line_angle_deg`,
  `line_cx` and `line_cy` (the line frame), `angle_source`, `pitch_px`, `model_offset_pitch`, `n_signs`, `signs_visible`, `sign_share`, `atf_match`, `n_outliers`, `order_ok`, `flags`, `split`
  and `tablet_level`. The `model_offset_pitch` column gives the distance to the nearest line found by the line model and is blank if there is none or the check was not run.
- `signs.csv` has one row per sign, in reading order. Its columns are `sign_id`, `line_id`, `face`, `tablet`, `pos` (1 is the first sign), `sign_name`, `reading`, `type`,
  `ebl_label`, `ebl_path` (line, word and sign), `face_x0..face_y1` (the box in the face photograph), `crop_x0..crop_y1` (the bounds of the box in the line image) and `outlier`.
- The folder `lines/<face>/` holds the line images, one `<line_id>.jpg` per line.

## Coordinate frames

The line frame is the face rotated with `cv2.getRotationMatrix2D((line_cx, line_cy), line_angle_deg, 1.0)`, at the same size. A line image's pixels are that frame minus the line's (`x0`, `y0`).

## Choosing lines

- The `tablet_level` column takes the values `full`, `all_lines`, `edges_missing` and `partial`. A `full` tablet has boxes on every transliterated line, with at least 80% of its visible signs boxed.
  Lines of a `partial` tablet can themselves be well annotated, so `sign_share` is the better guide for a single line.
- The `sign_share` column is the number of boxes on the line divided by the number of visible signs in `atf_text`, capped at 1, and it is blank without a transliteration. It is only an estimate, because the signs are counted by splitting the text.
  It counts boxes and does not check that they match the text, so a line whose `atf_text` belongs to another line can still score 1.
- The `atf_match` column is the share of the line's box readings that occur in `atf_text` (broken parts included, each word of the text used once), and it is blank
  without a transliteration. It is near 1 for most lines and near 0 where eBL labelled a line's boxes with a neighbouring line's number, so `atf_text` is the wrong line's.
- The `flags` column lists any of these problems:

  - `few_signs`: the line has fewer than 3 signs.
  - `no_atf`: no transliteration line was found.
  - `atf_mismatch`: fewer than half the box readings occur in `atf_text`, which usually means it is another line's transliteration.
  - `angle_off`: the truth segment is more than 5 degrees from the face's usual line angle, which often means a sign box was assigned to the wrong line.
  - `order`: the signs do not read left to right.
  - `outlier`: a sign lies far from the line.
  - `clipped`: the rectangle reaches the edge of the photograph.
  - `model_disagrees`: the line model found no line within 0.35 spacings, which often means a misplaced box or a damaged line. The model was trained on about half these faces, so it agrees more often there.

  A line with none of these flags and a `sign_share` of at least 0.8 is a good default, and there are 5304 such lines here.

## Caveats

- Neo-Assyrian is thinly annotated, and most of its tablets have boxes on only a few lines.
- eBL boxes are loose, and neighbouring lines' boxes overlap, so a line image also shows parts of its neighbours.
- The lines come from eBL's labels and are straight. They are occasionally wrong where a box was assigned to the wrong line.
- The `split` column is the line model's split, so a face marked `tune` may have been used to train it. Multi-column faces have no split.

## Licence

The photographs and sign annotations come from the Electronic Babylonian Library (<https://www.ebl.lmu.de/>), and the photographs were taken by the holding museums. British Museum images
are licensed CC BY-NC-SA 4.0, and the licences of others vary. The sign annotations are also published as the eBL sign dataset (Lewenstein et al. 2026, <https://doi.org/10.5334/johd.503>; CC BY-NC 4.0).
The set is for research use. Check the holding institution's terms before redistributing the images.
