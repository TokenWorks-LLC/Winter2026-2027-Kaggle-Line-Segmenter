"""Image preparation for running the line model on a new data set (segment/segment_images.py).

The model was trained on colour photographs of tablet faces, resized to 1024 pixels on the long side. New images can differ
in format, so load_image brings any of them to 8-bit, three-channel BGR, with the same pixel grid as the file on disk, so
that every coordinate the pipeline writes is in the pixels of the image as provided:
  - EXIF orientation is applied only with --exif (most data sets give coordinates in the stored pixel grid);
  - an alpha channel (for example MSII or other 3D renderings with a transparent background) is flattened onto a plain
    background, black by default, like the studio background of most eBL and CDLI photographs;
  - 16-bit images are scaled to 8 bits, and greyscale images are copied into three channels;
  - optionally, contrast is equalised with CLAHE (--clahe); on the Ur III MSII renderings it did not help.
find_views splits a photograph that shows several views of a tablet (a CDLI fat cross, an eBL composite) into one box per
view and guesses each view's role (label_views), including 'other' for scale bars and label cards; with views = 'whole'
the image is used as it is.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

from line_segmentation.segment.views import detect_views

Image.MAX_IMAGE_PIXELS = None
BACKGROUNDS = {'black': (0, 0, 0), 'white': (255, 255, 255), 'grey': (128, 128, 128)}


def load_image(path, background='black', exif=False, clahe=False):
    """Read any common image file and return (BGR uint8 image, notes on what was done)."""
    notes = []
    pil = Image.open(path)
    if exif:
        t = ImageOps.exif_transpose(pil)
        if t.size != pil.size or t is not pil: notes.append('exif orientation applied')
        pil = t
    if pil.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        a = np.asarray(pil, dtype=np.float64); a = (a / max(1.0, a.max()) * 255).astype(np.uint8); pil = Image.fromarray(a); notes.append('16-bit scaled to 8-bit')
    if pil.mode in ('RGBA', 'LA', 'PA') or (pil.mode == 'P' and 'transparency' in pil.info):
        rgba = np.asarray(pil.convert('RGBA'), dtype=np.float64); alpha = rgba[:, :, 3:4] / 255.0
        bg = np.array(BACKGROUNDS[background], dtype=np.float64)[::-1][None, None, :]       # BGR order below
        rgb = rgba[:, :, :3][:, :, ::-1]                                                     # to BGR
        img = (rgb * alpha + bg * (1 - alpha)).round().astype(np.uint8); notes.append(f'alpha flattened onto {background}')
    else:
        if pil.mode != 'RGB':
            notes.append(f'{pil.mode} converted to RGB')
        img = cv2.cvtColor(np.asarray(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); notes.append('CLAHE contrast')
    return np.ascontiguousarray(img), notes


def view_tint(img, box, bg_ab, delta=12.0, work=300):
    """Two cues for telling a tablet view from a scale bar or label card, on the view at about work pixels:
    tint, the share of pixels whose colour (LAB a, b) differs from the background's by more than delta (clay is tinted;
    print on white or grey stock is as neutral as the paper), and mid, the share of mid-tone pixels (between 35 and 80
    percent of the view's bright level; clay is mostly mid-tone, a card is bright stock with dark print)."""
    x0, y0, x1, y1 = box; v = img[y0:y1, x0:x1]
    if v.size == 0:
        return 0.0, 0.0
    s = min(1.0, work / max(v.shape[:2])); v = cv2.resize(v, (max(1, int(v.shape[1] * s)), max(1, int(v.shape[0] * s))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(v, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0]; hi = np.percentile(L, 95)
    tint = float((np.linalg.norm(lab[..., 1:] - bg_ab, axis=2) > delta).mean())
    mid = float(((L > 0.35 * hi) & (L < 0.8 * hi)).mean())
    return tint, mid


def background_ab(img, b_frac=0.01):
    """Background colour (LAB a, b) from the image border."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); b = max(2, int(b_frac * max(img.shape[:2])))
    e = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    return np.median(e[:, 1:].astype(np.float32), axis=0)


def label_views(views, edge_area=0.35, edge_aspect=2.2, tinted_min=0.12, other_rel=0.25, ref_area=0.25, mid_max=0.3):
    """Guess each view's role, in place. Roles are estimates.
    A view is 'other' (a scale bar, label card or the like, never segmented) when both cues of view_tint say so: its tint
    is under other_rel of the photo's most tinted large view (large = at least ref_area of the largest view, so a small
    coloured logo or colour chart does not set the reference; and only when that view's tint is at least tinted_min), and
    under mid_max of it is mid-tone. Tint is compared within one photo, so a photo whose tablets are all grey keeps every
    view; the mid-tone cue keeps grey-brown clay whose tint varies from view to view.
    Of the remaining views, the two largest that are neither small nor elongated are the obverse and reverse; the upper of
    the two (CDLI fat crosses and eBL composites put the obverse first) is the obverse. A view smaller than edge_area of the
    largest remaining view, or with sides in a ratio above edge_aspect, is an edge: top, bottom, left or right by where it
    lies relative to the obverse, else 'edge'. When no view qualifies (a photo of one long fragment, say), the most tinted
    remaining view is the obverse (the largest, without tints)."""
    if not views:
        return views
    area = lambda v: (v['box'][2] - v['box'][0]) * (v['box'][3] - v['box'][1])
    largest = max(area(v) for v in views)
    ref = max([v['tint'] for v in views if 'tint' in v and area(v) >= ref_area * largest], default=0.0)
    if ref >= tinted_min:
        for v in views:
            if v.get('tint', 1.0) < other_rel * ref and v.get('mid', 1.0) < mid_max: v['role'] = 'other'
    rest = [v for v in views if 'role' not in v]
    if not rest:
        return views
    big = max(area(v) for v in rest)
    def is_face(v):
        w, h = v['box'][2] - v['box'][0], v['box'][3] - v['box'][1]
        return area(v) >= edge_area * big and max(w, h) / max(1, min(w, h)) <= edge_aspect
    faces = sorted([v for v in rest if is_face(v)], key=lambda v: -area(v))[:2] or [max(rest, key=lambda v: (v.get('tint', 0.0), area(v)))]
    faces.sort(key=lambda v: (v['box'][1] + v['box'][3]) / 2)
    for v, role in zip(faces, ('obverse', 'reverse')):
        v['role'] = role
    ob = faces[0]['box']
    for v in rest:
        if 'role' in v:
            continue
        v['role'] = 'edge'
        cx, cy = (v['box'][0] + v['box'][2]) / 2, (v['box'][1] + v['box'][3]) / 2
        ocx, ocy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
        if ob[0] <= cx <= ob[2]: v['role'] = 'edge_top' if cy < ocy else 'edge_bottom'
        elif ob[1] <= cy <= ob[3]: v['role'] = 'edge_left' if cx < ocx else 'edge_right'
    return views


def find_views(img, views='whole', pad_frac=0.02):
    """Boxes [x0, y0, x1, y1] in image pixels to segment: the whole image, or each detected tablet view (largest first)."""
    H, W = img.shape[:2]
    if views == 'whole':
        return [dict(id='whole', box=[0, 0, W, H])]
    out = []; bg_ab = background_ab(img)
    for v in detect_views(img):
        x0, y0, x1, y1 = v['box']; pad = int(pad_frac * max(x1 - x0, y1 - y0))
        tint, mid = view_tint(img, v['box'], bg_ab)
        out.append(dict(id=f"v{v['id']}", box=[max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)], tint=tint, mid=mid))
    return label_views(out)
