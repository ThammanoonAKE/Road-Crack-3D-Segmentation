"""Difix3D-style fine-tuning of one road hole's splat (the cabinet B v2 recipe, difix_ft.py, for the road).

    python -I difix_road.py render CLIP CKPT ROUND OUT_DIR      novel views -> OUT_DIR/{render,alpha,ref}
    python -I difix_road.py train CLIP CKPT STEPS OUT_CKPT NOVEL_DIR [NOVEL_DIR ...]
    python -I difix_road.py eval CLIP CKPT_BEFORE CKPT_AFTER OUT_PREFIX
    python -I difix_road.py ply CLIP CKPT OUT_PLY

The Runner is rebuilt from the v2 run's own cfg.yml (and its depth prior), so the real frames, bilateral grids
and checkpoint line up. Novel views draw only the Gaussians inside the web crop (export_web_splat.py --crop
0.6 --max-height 0.2), on black. Each round turns 48 photo cameras about their aim points by ROUND x STEP_DEG
round the road's normal, either way, and adds an orbit of the crack at every 30 degrees, two heights, past
the side the phone never filmed. Difix (difix_fix.py, own venv) repairs each with its photo cut to the crop.
Training as in difix_ft.py: a repaired view on 30% of steps at 0.2 weight, scored over the whole frame
against Difix's output near the drawn road and black beyond it.
"""
import os
import random
import sys

import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository
# The work folders (road_crack\<clip> from clips_to_splats.ps1, difix\road_<clip> from these scripts)
DATA = os.environ.get("ROAD_CRACK_DATA") or rf"{ROOT}\data"
GSPLAT = rf"{ROOT}\third_party\gsplat\examples"
sys.path.insert(0, GSPLAT)
sys.path.insert(0, rf"{ROOT}\gaussian_splatting")
os.chdir(GSPLAT)

NW, NH = 1024, 576
NOVEL_P, NOVEL_W = 0.3, 0.2
STEP_DEG = 15
PHONE_HEIGHT, CROP, MAX_HEIGHT = 1.5, 0.6, 0.2  # build_all.ps1's export_web_splat.py settings
dev = "cuda"


def make_runner(clip, max_steps):
    os.environ["GSPLAT_DEPTH_PRIOR_DIR"] = rf"{DATA}\road_crack\{clip}\depth_prior"
    os.environ["GSPLAT_DEPTH_PRIOR_SAMPLES"] = "40000"  # clips_to_splats.ps1 -DepthPrior
    import simple_trainer
    from lib_bilagrid import BilateralGrid, slice
    from simple_trainer import Config, Runner

    with open(rf"{ROOT}\runs\road_crack_{clip}_v2\cfg.yml") as f:
        saved = yaml.unsafe_load(f)
    saved.update(result_dir=rf"{DATA}\difix\road_{clip}\runner", max_steps=max_steps,
                 disable_viewer=True, ckpt=None, resume=None)
    if os.environ.get("DIFIX_ROAD_SCENE"):  # e.g. difix_road_4k.py's 4K scene: same poses, other images
        saved["data_dir"] = os.environ["DIFIX_ROAD_SCENE"]
    cfg = Config(**saved)
    assert cfg.post_processing == "bilateral_grid" and not cfg.bilateral_grid_fused
    simple_trainer.BilateralGrid, simple_trainer.slice = BilateralGrid, slice
    return Runner(0, 0, 1, cfg)


def load(runner, path):
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]
    runner.post_processing_module.load_state_dict(ckpt["post_processing"])
    return ckpt["step"]


def save(runner, step, path):
    torch.save({"step": step, "splats": runner.splats.state_dict(),
                "post_processing": runner.post_processing_module.state_dict()}, path)


def web_frame(parser):
    """export_web_splat.py's frame: (middle, up, rot, scale, aims, phone distance, opening heading)."""
    from render_clip_tour import fit_plane

    c2ws = parser.camtoworlds[np.argsort(parser.image_names)]
    centre, up = fit_plane(parser.points, np.random.default_rng(0))
    if (c2ws[:, :3, 3].mean(0) - centre) @ up < 0:
        up = -up
    eyes, fwds = c2ws[:, :3, 3], c2ws[:, :3, 2]
    flat = lambda v: v - (v @ up)[..., None] * up  # noqa: E731
    dist = np.where(fwds @ up < -0.05, ((centre - eyes) @ up) / np.minimum(fwds @ up, -0.05), 3.0)
    aims = eyes + fwds * np.clip(dist, 0.3, 6.0)[:, None]
    phone_height = float(np.median((eyes - centre) @ up))
    gap = np.linalg.norm(flat(aims[:, None] - aims[None]), axis=2)
    near = gap[np.argmax((gap < 0.5 * phone_height).sum(1))] < 0.5 * phone_height
    middle = aims[near].mean(0)
    middle -= ((middle - centre) @ up) * up
    eyes_s = gaussian_filter1d(eyes, 3.6, axis=0, mode="nearest")
    aims_s = gaussian_filter1d(aims, 3.6, axis=0, mode="nearest")
    start = int(np.argmin(np.linalg.norm(flat(aims_s - middle), axis=1)))
    heading = flat((aims_s[start] - eyes_s[start])[None])[0]
    heading /= np.linalg.norm(heading)
    rot = np.stack([np.cross(heading, up), up, -heading])
    scale = PHONE_HEIGHT / phone_height
    reach = float(np.median(np.linalg.norm(eyes - aims, axis=1)))
    return middle, up, rot, scale, aims, reach, heading


def in_crop(runner):
    middle, up, rot, scale, aims, _, _ = web_frame(runner.parser)
    to_web = lambda p: (np.asarray(p, dtype=np.float64) - middle) @ rot.T * scale  # noqa: E731
    xyz = to_web(runner.splats["means"].detach().cpu().numpy())
    aim_web = to_web(aims)
    finite = np.isfinite(xyz).all(1)  # a few Gaussians in some runs have non-finite centres
    d = np.full(len(xyz), np.inf)
    d[finite], _ = cKDTree(aim_web[:, [0, 2]]).query(xyz[finite][:, [0, 2]], distance_upper_bound=CROP * PHONE_HEIGHT)
    keep = np.isfinite(d) & (np.abs(xyz[:, 1]) <= MAX_HEIGHT * PHONE_HEIGHT)
    if not finite.all():
        print(f"  {int((~finite).sum())} Gaussians with non-finite centres left out of the crop")
    return torch.as_tensor(keep, device=dev)


def crop_splats(runner, keep):
    s = dict(runner.splats.items())
    s["opacities"] = torch.where(keep, s["opacities"], torch.full_like(s["opacities"], -30.0))
    return s


def rot_about(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))[0]


def look_at(eye, target, up):
    """OpenCV camera-to-world (x right, y down, z forward) for a world up vector."""
    f = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, np.cross(f, x), f], 1)
    c2w[:3, 3] = eye
    return c2w


def draw(runner, splats, c2w, K, w=NW, h=NH, mode="RGB"):
    with torch.no_grad():
        kw = {"sh_degree": runner.cfg.sh_degree} if mode == "RGB" else {}
        img, alpha, _ = runner.rasterize_splats(torch.as_tensor(c2w, device=dev, dtype=torch.float32)[None],
                                                torch.as_tensor(K, device=dev, dtype=torch.float32)[None],
                                                w, h, splats=splats, near_plane=0.01, far_plane=1e10,
                                                render_mode=mode, **kw)
    return img[0], alpha[0, ..., 0]


def orbit_views(runner, K, els=(35, 55), back=1.3):
    """Cameras round the crack at every 30 degrees, looking at it from `back` x the phones' distance."""
    middle, up, _, _, _, reach, heading = web_frame(runner.parser)
    out = []
    for az in range(0, 360, 30):
        h = rot_about(up, az) @ -heading
        for el in els:
            eye = middle + reach * back * (np.cos(np.radians(el)) * h + np.sin(np.radians(el)) * up)
            out.append((f"az{az:03d}_el{el}", look_at(eye, middle, up), K))
    return out


def photo_K(ts):
    K = ts[0]["K"].numpy().astype(np.float64)
    K[:2] *= NW / ts[0]["image"].shape[1]
    return K


def render_round(clip, ckpt, rnd, out):
    for d in ("render", "alpha", "ref"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    runner = make_runner(clip, 100000)
    load(runner, ckpt)
    keep = in_crop(runner)
    splats = crop_splats(runner, keep)
    middle, up, _, _, _, _, _ = web_frame(runner.parser)
    ts = runner.trainset
    K = photo_K(ts)
    views = []  # (name, c2w, K, source photo index in the trainset)
    for i in sorted(random.Random(rnd).sample(range(len(ts)), min(48, len(ts)))):
        c2w = ts.parser.camtoworlds[ts.indices[i]].astype(np.float64)
        dep, a = draw(runner, splats, c2w, K, NW // 4, NH // 4, mode="ED")
        dep = dep[..., 0][a > 0.5]
        if dep.numel() < 500:
            continue
        pivot = c2w[:3, 3] + c2w[:3, 2] * float(dep.median())
        for sign in (-1, 1):
            R = rot_about(up, sign * STEP_DEG * rnd)
            n = np.eye(4)
            n[:3, :3] = R @ c2w[:3, :3]
            n[:3, 3] = pivot + R @ (c2w[:3, 3] - pivot)
            views.append((f"r{rnd}_c{i:03d}_{'l' if sign < 0 else 'r'}", n, K, i))
    cam_pos = np.array([ts.parser.camtoworlds[j][:3, 3] for j in ts.indices])
    rel = cam_pos - middle
    rel /= np.linalg.norm(rel, axis=1, keepdims=True)
    for name, c2w, Ko in orbit_views(runner, K):
        u = (c2w[:3, 3] - middle) / np.linalg.norm(c2w[:3, 3] - middle)
        views.append((f"r{rnd}_o_{name}", c2w, Ko, int(np.argmax(rel @ u))))
    for name, c2w, Kv, j in views:
        img, alpha = draw(runner, splats, c2w, Kv)
        cv2.imwrite(f"{out}/render/{name}.png", (img.clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8))
        cv2.imwrite(f"{out}/alpha/{name}.png", (alpha.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        _, ref_alpha = draw(runner, splats, ts.parser.camtoworlds[ts.indices[j]], K)
        photo = cv2.resize(cv2.imread(ts.parser.image_paths[ts.indices[j]]), (NW, NH), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out}/ref/{name}.png", (photo * ref_alpha.cpu().numpy()[..., None]).round().astype(np.uint8))
        np.save(f"{out}/render/{name}.npy", np.stack([c2w, np.pad(Kv, ((0, 1), (0, 1)))]).astype(np.float32))
    print(f"round {rnd}: {len(views)} novel views, {int(keep.sum()):,} of {len(keep):,} Gaussians in the web crop")


def novel_set(d):
    out = []
    for n in sorted(f[:-4] for f in os.listdir(f"{d}/render") if f.endswith(".npy")):
        cam = np.load(f"{d}/render/{n}.npy")
        fixed = cv2.imread(f"{d}/difix_ref/{n}.png")
        if fixed is None:
            continue
        a = cv2.imread(f"{d}/alpha/{n}.png", cv2.IMREAD_GRAYSCALE)
        near = cv2.dilate((a > 5).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        out.append((torch.as_tensor(cam[0], device=dev), torch.as_tensor(cam[1][:3, :3], device=dev),
                    torch.as_tensor((fixed[..., ::-1] * near[..., None]).copy(), device=dev).float() / 255))
    return out


def train(clip, ckpt, steps, out_ckpt, novel_dirs):
    import torch.nn.functional as F
    from gsplat.losses import depth_l1_loss, ssim_loss, total_variation_loss

    start = torch.load(ckpt, map_location="cpu", weights_only=True)["step"] + 1
    runner = make_runner(clip, start + steps)
    load(runner, ckpt)
    cfg = runner.cfg
    keep = in_crop(runner)
    novel = [novel_set(d) for d in novel_dirs]
    newest, older = novel[-1], [v for s in novel[:-1] for v in s]
    print(f"fine-tuning {steps} steps from {start}: {len(newest)} newest novel views, {len(older)} older")
    sched = [torch.optim.lr_scheduler.ExponentialLR(runner.optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)),
             torch.optim.lr_scheduler.ChainedScheduler([
                 torch.optim.lr_scheduler.LinearLR(runner.post_processing_optimizers[0], start_factor=0.01, total_iters=1000),
                 torch.optim.lr_scheduler.ExponentialLR(runner.post_processing_optimizers[0], gamma=0.01 ** (1.0 / cfg.max_steps))])]
    for _ in range(start):
        for s in sched:
            s.step()
    loader = torch.utils.data.DataLoader(runner.trainset, batch_size=1, shuffle=True, num_workers=4,
                                         persistent_workers=True, pin_memory=True)
    it = iter(loader)
    stats = {"real": [], "novel": []}
    for step in range(start, start + steps):
        if random.random() < NOVEL_P:
            c2w, K, target = random.choice(newest if (not older or random.random() < 0.7) else older)
            colors, _, _ = runner.rasterize_splats(c2w[None], K[None], NW, NH, splats=crop_splats(runner, keep),
                                                   sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane)
            l1 = (colors[0] - target).abs().mean()
            loss = NOVEL_W * torch.lerp(l1, ssim_loss(colors[0].permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]),
                                        cfg.ssim_lambda)
            stats["novel"].append(l1.item())
        else:
            try:
                data = next(it)
            except StopIteration:
                it = iter(loader)
                data = next(it)
            c2w, K = data["camtoworld"].to(dev), data["K"].to(dev)
            pixels = data["image"].to(dev) / 255.0
            H, W = pixels.shape[1:3]
            ids = data["image_id"].to(dev)
            renders, _, _ = runner.rasterize_splats(c2w, K, W, H, sh_degree=cfg.sh_degree, near_plane=cfg.near_plane,
                                                    far_plane=cfg.far_plane, image_ids=ids, render_mode="RGB+ED",
                                                    frame_idcs=ids)
            colors, depths = renders[..., :3], renders[..., 3:4]
            l1 = (colors - pixels).abs().mean()
            loss = torch.lerp(l1, ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)), cfg.ssim_lambda)
            pts = data["points"].to(dev)
            if pts.shape[1]:
                grid = torch.stack([pts[..., 0] / (W - 1) * 2 - 1, pts[..., 1] / (H - 1) * 2 - 1], -1).unsqueeze(2)
                d = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True).squeeze(3).squeeze(1)
                loss = loss + cfg.depth_lambda * depth_l1_loss(d, data["depths"].to(dev), scene_scale=runner.scene_scale)
            loss = loss + 10 * total_variation_loss(runner.post_processing_module.grids)
            stats["real"].append(l1.item())
        loss.backward()
        for opt in list(runner.optimizers.values()) + list(runner.post_processing_optimizers):
            opt.step()
            opt.zero_grad(set_to_none=True)
        for s in sched:
            s.step()
        if (step - start + 1) % 500 == 0:
            print(f"  step {step + 1}: real L1 {np.mean(stats['real'][-300:]):.4f}, "
                  f"novel L1 {np.mean(stats['novel'][-150:] or [0]):.4f}", flush=True)
    save(runner, start + steps - 1, out_ckpt)


def evaluate(clip, before, after, prefix):
    """Held-out photos (the run's every-8th val split), web crop only, per-frame colour fit, inside where both
    models cover the frame; a 1:1 crop sheet photo | before | after; and an orbit sheet before | after."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    runner = make_runner(clip, 100000)
    lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    models = {}
    for tag, path in (("before", before), ("after", after)):
        load(runner, path)
        keep = in_crop(runner)
        models[tag] = {k: v.detach().clone() for k, v in crop_splats(runner, keep).items()}
    vs = runner.valset
    scores = {k: {"psnr": [], "lpips": []} for k in models}
    crops = []
    for n in range(len(vs)):
        data = vs[n]
        photo = data["image"].to(dev).float() / 255
        H, W = photo.shape[:2]
        c2w, K = data["camtoworld"].numpy(), data["K"].numpy()
        out = {k: draw(runner, s, c2w, K, W, H) for k, s in models.items()}
        mask = (out["before"][1] > 0.9) & (out["after"][1] > 0.9)
        if mask.float().mean() < 0.1:
            continue
        fitted = {}
        for k, (img, _) in out.items():
            A = torch.cat([img[mask], torch.ones_like(img[mask][:, :1])], 1).double()
            fit = torch.linalg.solve(A.T @ A, A.T @ photo[mask].double()).float()
            o = photo.clone()
            o[mask] = (A.float() @ fit).clamp(0, 1)
            fitted[k] = o
            scores[k]["psnr"].append((-10 * torch.log10(((o - photo) ** 2)[mask].mean())).item())
            scores[k]["lpips"].append(lp(o.permute(2, 0, 1)[None], photo.permute(2, 0, 1)[None]).item())
        if len(crops) < 4:
            ys, xs = torch.nonzero(mask, as_tuple=True)
            cy, cx = int(ys.float().median()), int(xs.float().median())
            y0, x0 = max(0, min(H - 400, cy - 200)), max(0, min(W - 560, cx - 280))
            row = [(t[y0:y0 + 400, x0:x0 + 560].cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
                   for t in (photo, fitted["before"], fitted["after"])]
            for t, lab in zip(row, ("photo", "before", "after Difix")):
                cv2.putText(t, lab, (8, 26), 0, 0.8, (0, 0, 255), 2)
            crops.append(np.hstack(row))
    for k, v in scores.items():
        print(f"  {k:7s} PSNR {np.median(v['psnr']):.2f} dB, LPIPS {np.median(v['lpips']):.3f} on {len(v['psnr'])} held-out photos")
    a, b = np.array(scores["before"]["psnr"]), np.array(scores["after"]["psnr"])
    la, lb = np.array(scores["before"]["lpips"]), np.array(scores["after"]["lpips"])
    print(f"  after better: PSNR on {(b > a).sum()} of {len(a)}, LPIPS on {(lb < la).sum()} of {len(la)}")
    cv2.imwrite(f"{prefix}_crops.jpg", np.vstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 92])
    rows = []
    for name, c2w, Ko in orbit_views(runner, photo_K(runner.trainset), els=(20, 40), back=1.6)[::3]:
        t = [(draw(runner, s, c2w, Ko)[0].clamp(0, 1).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
             for s in models.values()]
        for im, lab in zip(t, models):
            cv2.putText(im, f"{name} {lab}", (8, 28), 0, 0.8, (0, 0, 255), 2)
        rows.append(np.hstack(t))
    cv2.imwrite(f"{prefix}_orbit.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def ply(clip, ckpt, out):
    from gsplat import export_splats
    runner = make_runner(clip, 100000)
    load(runner, ckpt)
    s = runner.splats
    export_splats(means=s["means"], scales=s["scales"], quats=s["quats"], opacities=s["opacities"], sh0=s["sh0"],
                  shN=s["shN"], format="ply", save_to=out)


if __name__ == "__main__":
    mode, clip = sys.argv[1], sys.argv[2]
    if mode == "render":
        render_round(clip, sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif mode == "train":
        train(clip, sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6:])
    elif mode == "eval":
        evaluate(clip, sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "ply":
        ply(clip, sys.argv[3], sys.argv[4])
