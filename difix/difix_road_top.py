"""How steeply did the phone look down, and what does the crack look like from straight above?

    python -I difix_road_top.py CLIP CKPT_BEFORE CKPT_AFTER OUT_JPG

Prints the photos' downward tilt (90 = straight down) and their elevation seen from the crack, then renders
the web crop from above, before | after: the whole patch (1.6x the phones' distance) on the first row, and
a close view at the phones' distance on the second, both at the photos' resolution and lens.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from difix_road import crop_splats, draw, in_crop, load, look_at, make_runner, web_frame  # noqa: E402


def main(clip, before, after, out):
    runner = make_runner(clip, 100000)
    middle, up, _, _, _, reach, heading = web_frame(runner.parser)
    c2ws = runner.parser.camtoworlds
    tilt = np.degrees(np.arcsin(np.clip(-(c2ws[:, :3, 2] @ up), -1, 1)))
    rel = c2ws[:, :3, 3] - middle
    elev = np.degrees(np.arcsin(np.clip((rel @ up) / np.linalg.norm(rel, axis=1), -1, 1)))
    pct = lambda a: " / ".join(f"{v:.0f}" for v in np.percentile(a, [0, 10, 50, 90, 100]))  # noqa: E731
    print(f"photo tilt below horizon (min / 10% / median / 90% / max): {pct(tilt)} deg")
    print(f"photo elevation seen from the crack:                       {pct(elev)} deg")
    ts = runner.trainset
    K = ts[0]["K"].numpy().astype(np.float64)
    H, W = ts[0]["image"].shape[:2]
    tiles = {}
    for tag, path in (("before", before), ("after Difix", after)):
        load(runner, path)
        splats = crop_splats(runner, in_crop(runner))
        for back in (1.6, 1.0):
            # straight down, the far side of the opening view at the top of the frame
            img, _ = draw(runner, splats, look_at(middle + up * reach * back, middle, heading), K, W, H)
            t = (img.clamp(0, 1).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
            cv2.putText(t, f"top, {back}x distance, {tag}", (12, 40), 0, 1.2, (0, 0, 255), 3)
            tiles[(back, tag)] = t
    rows = [np.hstack([tiles[(b, "before")], tiles[(b, "after Difix")]]) for b in (1.6, 1.0)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {out} ({W}x{H} per view)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
