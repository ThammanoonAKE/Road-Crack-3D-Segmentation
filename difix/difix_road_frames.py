"""Per-frame look at difix_road.py's held-out score: which photos got worse, and where in the frame.

    python -I difix_road_frames.py CLIP CKPT_BEFORE CKPT_AFTER OUT_PREFIX

Prints PSNR/LPIPS per held-out photo (same colour fit and mask as difix_road.evaluate) and how much of the
before/after change lies within 40 px of the mask's edge. Writes OUT_PREFIX_worst.jpg for the photo whose LPIPS
rose most: photo | before | after | |after - before| x4, half size, then a 1:1 crop where they differ most.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from difix_road import crop_splats, dev, draw, in_crop, load, make_runner  # noqa: E402


def main(clip, before, after, prefix):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    runner = make_runner(clip, 100000)
    lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    models = {}
    for tag, path in (("before", before), ("after", after)):
        load(runner, path)
        models[tag] = {k: v.detach().clone() for k, v in crop_splats(runner, in_crop(runner)).items()}
    vs = runner.valset
    rows, worst = [], None
    for n in range(len(vs)):
        data = vs[n]
        photo = data["image"].to(dev).float() / 255
        H, W = photo.shape[:2]
        c2w, K = data["camtoworld"].numpy(), data["K"].numpy()
        out = {k: draw(runner, s, c2w, K, W, H) for k, s in models.items()}
        mask = (out["before"][1] > 0.9) & (out["after"][1] > 0.9)
        if mask.float().mean() < 0.1:
            continue
        fitted, sc = {}, {}
        for k, (img, _) in out.items():
            A = torch.cat([img[mask], torch.ones_like(img[mask][:, :1])], 1).double()
            fit = torch.linalg.solve(A.T @ A, A.T @ photo[mask].double()).float()
            o = photo.clone()
            o[mask] = (A.float() @ fit).clamp(0, 1)
            fitted[k] = o
            sc[k] = ((-10 * torch.log10(((o - photo) ** 2)[mask].mean())).item(),
                     lp(o.permute(2, 0, 1)[None], photo.permute(2, 0, 1)[None]).item())
        m = mask.cpu().numpy().astype(np.uint8)
        edge = (m - cv2.erode(m, np.ones((81, 81), np.uint8))).astype(bool)
        diff = (fitted["after"] - fitted["before"]).abs().mean(2).cpu().numpy() * m
        at_edge = diff[edge].sum() / max(diff.sum(), 1e-9)
        rows.append((n, sc["before"], sc["after"], float(m.mean()), at_edge))
        rise = sc["after"][1] - sc["before"][1]
        if worst is None or rise > worst[0]:
            worst = (rise, n, photo.cpu().numpy(), fitted["before"].cpu().numpy(), fitted["after"].cpu().numpy(), diff)
    print(f"{'frame':>5} {'PSNR before':>11} {'after':>6} {'LPIPS before':>12} {'after':>6} {'mask':>5} {'change at edge':>14}")
    for n, b, a, cover, at_edge in rows:
        flag = "  worse" if a[1] > b[1] else ""
        print(f"{n:5d} {b[0]:11.2f} {a[0]:6.2f} {b[1]:12.3f} {a[1]:6.3f} {cover:5.0%} {at_edge:14.0%}{flag}")
    _, n, photo, b, a, diff = worst
    heat = np.repeat(np.clip(diff * 4, 0, 1)[..., None], 3, 2)
    half = [cv2.resize((t[..., ::-1] * 255).astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            for t in (photo, b, a, heat)]
    for t, lab in zip(half, ("photo", "before", "after Difix", "|after-before| x4")):
        cv2.putText(t, lab, (8, 26), 0, 0.8, (0, 0, 255), 2)
    H, W = diff.shape
    cy, cx = np.unravel_index(np.argmax(cv2.blur(diff, (101, 101))), diff.shape)
    y0, x0 = max(0, min(H - 400, cy - 200)), max(0, min(W - 560, cx - 280))
    crop = [(t[y0:y0 + 400, x0:x0 + 560][..., ::-1] * 255).astype(np.uint8).copy() for t in (photo, b, a)]
    for t, lab in zip(crop, ("photo 1:1", "before 1:1", "after 1:1")):
        cv2.putText(t, lab, (8, 26), 0, 0.8, (0, 0, 255), 2)
    top = np.vstack([np.hstack(half[:2]), np.hstack(half[2:])])
    bottom = np.hstack(crop)
    pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), np.uint8) if top.shape[1] > bottom.shape[1] else None
    if pad is not None:
        bottom = np.hstack([bottom, pad])
    else:
        top = np.hstack([top, np.zeros((top.shape[0], bottom.shape[1] - top.shape[1], 3), np.uint8)])
    cv2.imwrite(f"{prefix}_worst.jpg", np.vstack([top, bottom]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"worst LPIPS rise: frame {n} (+{worst[0]:.3f}) -> {prefix}_worst.jpg, crop at x {x0} y {y0}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
