# Running the line model on a new data set

`segment_images.py` runs the current line model on any folder or list of tablet images and writes the lines it finds, as
polygons and boxes in the pixels of each image as provided. `preprocess.py` prepares the images. Run everything from the
repository root with the project's virtual environment.

## Quick start

For a folder of images, with every image treated as one tablet side:

```
.venv/bin/python -m line_segmentation.segment.segment_images --images <folder> --out <output folder> --overlays 20
```

For a CSV that lists the images, with an id column and a file column:

```
.venv/bin/python -m line_segmentation.segment.segment_images \
    --list <images.csv> --id-col side_id --file-col file --out <output folder> --overlays 20 --crops
```

To try it on the example photographs in this repository:

```
.venv/bin/python -m line_segmentation.segment.segment_images --images examples/images --out output/example --crops --overlays 5
```

In `--list` mode, paths in `--file-col` are relative to the CSV's folder unless `--root` says otherwise. In `--images`
mode, the folder is searched recursively for PNG, JPEG, TIFF, BMP and WebP files, and each image's side id is its path
without the extension.

## What the pipeline does

1. **Preprocessing** (`preprocess.load_image`). Each image becomes an 8-bit, three-channel image on the same pixel grid as
   the file, so every coordinate written refers to the image as provided. A transparent background, as in MSII and other
   3D renderings, is flattened onto `--background` (black by default, like most museum photographs). 16-bit images are
   scaled to 8 bits, and greyscale images are copied into three channels. `--clahe` equalises the contrast, and `--exif`
   applies the EXIF orientation; both are off by default.
2. **Views** (`preprocess.find_views`). With `--views whole` (the default) each image is one side. With `--views auto` a
   photograph that shows several views of a tablet, such as a CDLI fat cross or an eBL composite, is split into one image
   per view, using the background detection in `views.py`. On a dark background, tablet material is whatever is brighter
   than the cloth. On a white or light grey background, a pixel counts as tablet when its colour differs from the paper's,
   when it is much darker than the paper at that point, or when it is somewhat darker and textured. The paper's brightness
   is modelled as a smooth surface, so uneven lighting and soft drop shadows do not join neighbouring views into one; with a
   plain brightness threshold they did. Each view is saved as `views/<side id>__v<k>.png`
   (largest view first, with a small margin around it) and is segmented as a side of its own. Each view also gets an
   estimated role (`preprocess.label_views`). Scale bars and label cards become `other`, saved but never segmented: a view
   is `other` when it is far less tinted than the photo's tablet views (under a quarter of the most tinted view at least a
   quarter the size of the largest) and under 30 percent of it is mid-tone (print on card stock, not clay). The tint is
   compared within one photo, so on a photo whose tablets are all grey nothing is set aside. Of the remaining views, the
   two largest that are neither small (under 35 percent of the largest) nor elongated (sides in a ratio above 2.2) are the
   obverse and the reverse, the upper one being the obverse, as CDLI fat crosses and eBL composites lay them out. When no
   view qualifies, as on a photo of one long fragment, the most tinted view is the obverse. Every other view is an edge,
   named `edge_top`, `edge_bottom`, `edge_left` or `edge_right` by its place next to the obverse, or plain `edge` when it is
   not next to it. With `--edges skip` (the
   default), edges are saved but not segmented, because the model finds false lines on most uninscribed edges; on 30 Ur III
   CDLI fat crosses, it put a line on 73 percent of edges, while the transliterations show writing on only 5 of 126. Use
   `--edges segment` to run the model on edges as well.
3. **Line model.** The model (`--weights`, by default `paths.LINE_WEIGHTS`, currently `run6_heights_s1`) runs on each side
   resized to 1024 pixels on the long side. The read-out (`linemodel/infer.read_lines`) uses the settings stored in the
   run's `config.json`; every setting can be overridden with options such as `--kernel-thr`.
   **Orientation.** The model only finds lines that run roughly across the image, so a side photographed on its side
   (text lines running up and down) gets a few false, low-scoring lines. With `--orient auto`, the model also runs on each
   side turned 90 degrees clockwise and anticlockwise, and a turn is kept when its mean line score beats the unturned
   side's by more than `--orient-margin` (0.03). Coordinates are still written in the pixels of the image as provided;
   the line order and crops follow the turned side, so line 1 is the top line of the text as read. `lines.json` records
   `rotation_deg` and `orient_gap` (best turned score minus unturned score) for each side, and `summary.csv` has both as
   columns. On 298 Old Assyrian obverses and reverses, 131 were turned; on a sideways side the turn raised the mean
   score from about 0.57 to 0.72–0.78. Sides with a gap near the margin are close calls worth checking in the overlays.
   It cannot tell upright from upside down (both give level lines), and it triples the model's running time.
4. **Line regions.** Each predicted line becomes a four-cornered region. Its top and bottom are the middle of the gaps to
   the neighbouring lines, extended by `--extend` line spacings (0.2) and moved down by `--shift` spacings (0.06), the crop
   rule calibrated on eBL faces. The region runs `--end-margin` spacings (0.5) past the line's ends and follows the line's
   own angle when it is within `--max-line-dev` degrees (20) of the side's usual angle.
5. **Output.** The pipeline writes the files described below. Lines are ordered top to bottom, every side appears even
   when no line was found, and the output is the model's raw read-out: nothing is corrected by hand, and low-scoring lines
   are kept.

## Output

- `lines.json` has three parts.
  - `segmenter` gives the model's name, the git commit and the weights file with its checksum (`version_or_commit`), and
    the tablets the weights were trained or validated on (`trained_on`, with per-face detail in `trained_on_detail`). It
    also says what the read-out and region rule were tuned on (`tuning`).
  - `settings` records every preprocessing and read-out setting used.
  - `sides` has one entry per side, with `side_id`, `width`, `height` and `lines`. With `--views auto`, a side also has
    `image` (its saved view image), `role` (its estimated role), `skipped` (when it is an edge that was not segmented),
    `source_image` and `source_file` (the photograph it came from) and `view_box`
    (`[x0, y0, x1, y1]`, where the view sits in that photograph). Its coordinates are then in the pixels of the view image;
    add `view_box`'s `x0` and `y0` to place them in the photograph. Each line has `order` (1 is the top
    line), `polygon` (four corners, clockwise from the top left of the line), `bbox` (`[x0, y0, x1, y1]`, clipped to the
    image), `score` (the mean kernel probability, between 0 and 1), `baseline` (the predicted centre line's two ends) and
    `angle_deg`. With `--crops`, each line also names its crop image.
- `summary.csv` has one row per side, with the turn applied and its score gap (with `--orient auto`), the number of lines, the mean score, the time taken and the preprocessing notes.
- `overlays/<side>.jpg` (with `--overlays N`) draws the regions in orange, the centre lines in red, and each line's order
  and score on the first N images.
- `crops/<side>__L<nn>.jpg` (with `--crops`) holds one image per line, cut along the line's own angle.
- `views/<side>__v<k>.png` (with `--views auto`) holds one image per detected view.

## Training data and leakage

`trained_on` comes from `tablet_data/line_model_training_faces.csv`, which `linemodel/list_training_faces.py` writes. If
that file is missing, run the script first; `trained_on` is otherwise marked unknown. Before sharing results on a data set,
compare its tablets with `trained_on` by museum number and by CDLI P-number, so that no test tablet was seen in training.

## What to expect on new material

The model was trained on photographs of Old Babylonian, Neo-Assyrian, Neo-Babylonian and Old Assyrian tablets. On a set
of 3D (MSII) renderings of Ur III tablets, it finds the lines of clean, well-preserved sides well. It also reads the inscriptions of seal
impressions as lines, which the transliteration lists separately from the side's own lines, and it finds few lines on
badly worn surfaces. Line counts on the Ur III sides matched the CDLI transliteration exactly on 22 percent of sides and
within one line on 62 percent, mostly because of seal impressions and lines that run onto the edges. On CDLI fat-cross photographs
with `--views auto --edges segment`, the model also finds one or two false lines on most uninscribed edges; segmenting edges
at the obverse's scale instead of enlarging them did not prevent this. Roles are estimates from the layout, and other parts
of a photograph, such as a museum logo, can be detected as views and labelled `edge`. Views that touch or nearly touch in the
photograph (gaps of a few pixels) can come out as one view, more often on light backgrounds than on dark ones. On 263 Middle
Assyrian fragment photographs (one fragment, a scale bar and a label card on white or light grey paper), `--views auto` gave
every photograph an obverse on the fragment; before scale bars were set aside and long fragments allowed as the obverse,
50 of them had none. Check the overlays before relying on a new data set.

On light backgrounds the model itself needs no special treatment: on 120 eBL test faces moved from black cloth onto a synthetic
light grey background with uneven lighting and drop shadows, recall fell from 0.94 to 0.92 and precision from 0.89 to 0.88.
Painting the background black before running the model did not help (recall 0.88), so the pipeline leaves the image as it is.
