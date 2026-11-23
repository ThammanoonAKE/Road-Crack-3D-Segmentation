"""Top-down novel views of one road hole for Difix (difix_road.py's render, but from above the hole).

    python -I difix_road_topviews.py CLIP CKPT ROUND OUT_DIR      -> OUT_DIR/{render,alpha,ref}

The phone looked down 30-63 degrees, so difix_road.py's rounds (photo tilt kept, orbits at 35 and 55) never
saw the crack from above. Here each round looks down a little steeper: 68, 79, then 90 degrees. The views
stay over the hole: 16 centres spread over where the photos aimed (farthest-point sampled), each seen from
4 sides at the phones' distance and at 0.6 of it (about the photos' detail at 1024 px wide). Each view's
Difix guide is the photo that aimed nearest its centre, steeper photos preferred, cut to the web crop.
Training, scoring and the web build are difix_road.py's.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from difix_road import NH, NW, crop_splats, draw, in_crop, load, look_at, make_runner, photo_K, rot_about, web_frame  # noqa: E402

ELEVATION = {1: 68, 2: 79, 3: 90}
CENTRES, BACKS, SIDES = 16, (1.0, 0.6), (0, 90, 180, 270)


def aim_of(c2w, middle, up):
    eye, fwd = c2w[:3, 3], c2w[:3, 2]
    dist = ((middle - eye) @ up) / min(fwd @ up, -0.05) if fwd @ up < -0.05 else 3.0
    return eye + fwd * np.clip(dist, 0.3, 6.0)


def render_top(clip, ckpt, rnd, out):
    for d in ("render", "alpha", "ref"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    runner = make_runner(clip, 100000)
    load(runner, ckpt)
    keep = in_crop(runner)
    splats = crop_splats(runner, keep)
    middle, up, _, _, _, reach, heading = web_frame(runner.parser)
    ts = runner.trainset
    K = photo_K(ts)
    c2ws = [ts.parser.camtoworlds[i].astype(np.float64) for i in ts.indices]
    aims = np.array([aim_of(c, middle, up) for c in c2ws])
    aims -= ((aims - middle) @ up)[:, None] * up
    tilt = np.array([-(c[:3, 2] @ up) for c in c2ws])  # sin of the photo's downward tilt
    centres = [aims[np.argmin(np.linalg.norm(aims - middle, axis=1))]]
    gap = np.linalg.norm(aims - centres[0], axis=1)
    while len(centres) < min(CENTRES, len(aims)):
        i = int(np.argmax(gap))
        centres.append(aims[i])
        gap = np.minimum(gap, np.linalg.norm(aims - aims[i], axis=1))
    el = np.radians(ELEVATION[rnd])
    views = []
    for ci, c in enumerate(centres):
        near = np.linalg.norm(aims - c, axis=1)
        ref = int(np.argmin(near / reach - 0.5 * tilt))  # close aim first, then the steeper photo
        for back in BACKS:
            for side in SIDES:
                h = rot_about(up, side) @ -heading  # from the centre toward the camera, flat
                eye = c + reach * back * (np.cos(el) * h + np.sin(el) * up)
                hint = up if ELEVATION[rnd] < 90 else -h  # straight down: the side's far edge at the top
                views.append((f"t{rnd}_c{ci:02d}_b{int(back * 10)}_s{side:03d}", look_at(eye, c, hint), ref))
    for name, c2w, j in views:
        img, alpha = draw(runner, splats, c2w, K)
        cv2.imwrite(f"{out}/render/{name}.png", (img.clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8))
        cv2.imwrite(f"{out}/alpha/{name}.png", (alpha.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        _, ref_alpha = draw(runner, splats, c2ws[j], K)
        photo = cv2.resize(cv2.imread(ts.parser.image_paths[ts.indices[j]]), (NW, NH), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out}/ref/{name}.png", (photo * ref_alpha.cpu().numpy()[..., None]).round().astype(np.uint8))
        np.save(f"{out}/render/{name}.npy", np.stack([c2w, np.pad(K, ((0, 1), (0, 1)))]).astype(np.float32))
    print(f"top round {rnd}: {len(views)} views looking down {ELEVATION[rnd]} degrees over {len(centres)} centres, "
          f"{int(keep.sum()):,} Gaussians in the web crop")


if __name__ == "__main__":
    render_top(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
