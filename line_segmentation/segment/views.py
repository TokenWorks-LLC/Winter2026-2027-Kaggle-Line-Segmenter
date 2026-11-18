"""Find tablet views in a photograph without annotations (moved from the old Line Splitter app).

The photo is thresholded against its background, connected bright regions are split at empty gaps, and every region
large enough to be a tablet face is offered as a view.
"""
import cv2
import numpy as np
from PIL import Image

from line_segmentation.data.faces.prepare_faces import tablet_components

Image.MAX_IMAGE_PIXELS = None


def _split_pieces(mask, box, min_gap=3, empty=0.04, depth=0):
    """Split a component box (in mask pixels) at runs of near-empty columns, then rows; return tight piece boxes."""
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1] > 0
    if not sub.any():
        return []
    for axis in (1, 0):
        prof = sub.mean(axis=0) if axis == 1 else sub.mean(axis=1)
        gaps = prof < empty
        cuts, i = [], 0
        while i < len(gaps):
            if gaps[i]:
                j = i
                while j < len(gaps) and gaps[j]:
                    j += 1
                if j - i >= min_gap and i > 0 and j < len(gaps):
                    cuts.append((i + j) // 2)
                i = j
            else:
                i += 1
        if cuts and depth < 6:
            edges = [0] + cuts + [len(gaps)]
            pieces = []
            for a, b in zip(edges, edges[1:]):
                nb = (x0 + a, y0, x0 + b, y1) if axis == 1 else (x0, y0 + a, x1, y0 + b)
                pieces += _split_pieces(mask, nb, min_gap, empty, depth + 1)
            return pieces
    cols = np.where(sub.any(axis=0))[0]
    rows = np.where(sub.any(axis=1))[0]
    return [(x0 + cols[0], y0 + rows[0], x0 + cols[-1] + 1, y0 + rows[-1] + 1)]


def detect_views(img_bgr, min_frac=0.01):
    """Candidate tablet views in a photo, as boxes in image pixels, largest first. Falls back to the whole image."""
    H, W = img_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    scale = max(2, int(round(max(W, H) / 1200)))
    comps, _lab, _ = tablet_components(pil, scale=scale)
    mask = tablet_components.mask
    views = []
    for _i, box, _a in comps:
        sb = tuple(int(round(v / scale)) for v in box)
        for p in _split_pieces(mask, sb):
            bx = (p[0] * scale, p[1] * scale, min(W, p[2] * scale), min(H, p[3] * scale))
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            if area >= min_frac * W * H and min(bx[2] - bx[0], bx[3] - bx[1]) >= 60:
                views.append(dict(box=[int(v) for v in bx], area=int(area)))
    views.sort(key=lambda v: -v['area'])
    if not views:
        views = [dict(box=[0, 0, W, H], area=W * H)]
    for k, v in enumerate(views):
        v['id'] = k
    return views
