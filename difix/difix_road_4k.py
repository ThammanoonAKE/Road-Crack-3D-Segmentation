"""4K detail for one road hole, in tiles. The phone filmed 4K, but the splat trained on 1920x1080 frames.

    python -I difix_road_4k.py train CLIP CKPT STEPS OUT_CKPT DENSIFY NOVEL_DIR [NOVEL_DIR ...]
    python -I difix_road_4k.py render CLIP CKPT OUT_DIR
    python -I difix_road_4k.py stitch OUT_DIR

Needs road_<clip>\\scene_4k: the clip's SfM undistorted at full size (colmap image_undistorter, no
max_image_size). Its poses are the 1080p scene's, so checkpoints carry over.

train: the Gaussians outside the web crop go first, leaving room for finer ones inside. Each real step trains a
random 1920x1080 tile of a 4K frame, the 1080p frames' size, scored only where the crop draws (alpha > 0.5)
less ERODE px at its edge. Without that margin the edge, darker than the road beyond it, was pulled outward
step after step, with the densifying feeding it, and both losses climbed (35000: real 0.0468 -> 0.0496, novel
0.0398 -> 0.0541 in 2000 steps). The bilateral grid and the depth prior are placed by the tile's position in
the whole frame. With DENSIFY=1
the default strategy splits and prunes over the first 60% of the steps (no opacity resets) until CAP
Gaussians. Novel views from the NOVEL_DIRs, difix_road's or this script's stitched ones, are mixed in as in
difix_road.train, the last dir taking 70% of those draws.

render: close views from above the hole, 79 and 90 degrees at 0.55 of the phones' distance, 16 centres x 4
sides, drawn at 2048x1152, about the 4K photos' detail. Each is cut into 3x3 overlapping 1024x576 tiles, the
size Difix works at. A tile's guide is the 4K photo that sees its middle best, reprojected into the tile through
the rendered depth, so Difix is shown the real detail. Where that photo is hidden or out of frame the guide
shows the render instead: black there came back from Difix as black pits that neither the model nor the
photo has.

stitch: the Difix'd tiles back into whole views, feathered across the overlaps; tiles too empty to repair
keep the render.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import difix_road as dr  # noqa: E402
from difix_road import NOVEL_P, NOVEL_W, crop_splats, dev, draw, in_crop, load, look_at, make_runner, photo_K, rot_about, save, web_frame  # noqa: E402
from difix_road_topviews import aim_of  # noqa: E402

TW, TH = 1920, 1080  # real-frame tile
# this machine has rebooted while densifying past ~9M Gaussians; with ~9 GB of the 24 held by other programs,
# 4.86M already took the card to 21 GB, so DIFIX_4K_CAP lowers it
CAP = int(os.environ.get("DIFIX_4K_CAP", 8_500_000))
RW, RH, TILE_W, TILE_H, STRIDE_X, STRIDE_Y = 2048, 1152, 1024, 576, 512, 288
ELEVATIONS, BACK, CENTRES, SIDES = (79, 90), 0.55, 16, (0, 90, 180, 270)
FEATHER = 192
ERODE = 33  # px of a 4K tile (about 16 at 1080p): wider than an edge Gaussian's footprint


def use_4k(clip):
    os.environ["DIFIX_ROAD_SCENE"] = rf"C:\3D-segmentation-data\difix\road_{clip}\scene_4k"


def load_views(d):
    """Novel views as (c2w, K, uint8 target) on the GPU: this script's stitched views (cam/, full/) or
    difix_road's (render/*.npy, difix_ref/). Black beyond 7 px of what the view drew."""
    hr = os.path.isdir(f"{d}/cam")
    cams = f"{d}/cam" if hr else f"{d}/render"
    out = []
    for n in sorted(f[:-4] for f in os.listdir(cams) if f.endswith(".npy")):
        cam = np.load(f"{cams}/{n}.npy")
        fixed = cv2.imread(f"{d}/{'full' if hr else 'difix_ref'}/{n}.png")
        if fixed is None:
            continue
        a = cv2.imread(f"{d}/alpha/{n}.png", cv2.IMREAD_GRAYSCALE)
        near = cv2.dilate((a > 5).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        out.append((torch.as_tensor(cam[0], device=dev), torch.as_tensor(cam[1][:3, :3], device=dev),
                    torch.as_tensor((fixed[..., ::-1] * near[..., None]).copy(), device=dev)))
    return out


def train(clip, ckpt, steps, out_ckpt, densify, novel_dirs):
    import torch.nn.functional as F
    from gsplat.losses import depth_l1_loss, ssim_loss, total_variation_loss
    from gsplat.strategy.ops import remove
    from lib_bilagrid import slice as bilagrid_slice

    use_4k(clip)
    start = torch.load(ckpt, map_location="cpu", weights_only=True)["step"] + 1
    runner = make_runner(clip, start + steps)
    load(runner, ckpt)
    cfg, st = runner.cfg, runner.cfg.strategy
    state = st.initialize_state(scene_scale=runner.scene_scale)
    keep = in_crop(runner)
    remove(params=runner.splats, optimizers=runner.optimizers, state=state, mask=~keep)
    print(f"{int(keep.sum()):,} of {len(keep):,} Gaussians kept, the web crop")
    scene = getattr(runner, "scene", None)
    if scene is not None:  # its per-Gaussian index kept the runner's initial count through load() and the crop
        assert not scene.signal, list(scene.signal)
        scene.component_index = torch.zeros(len(runner.splats["means"]), dtype=torch.long, device=dev)
    if densify:
        st.refine_start_iter, st.refine_stop_iter, st.reset_every = start + 100, start + int(steps * 0.6), 10 ** 9
    else:
        st.refine_stop_iter = 0
    novel = [load_views(d) for d in novel_dirs]
    newest, older = (novel[-1], [v for s in novel[:-1] for v in s]) if novel else ([], [])
    print(f"{steps} steps from {start} on 4K tiles{', densifying' if densify else ''}: "
          f"{len(newest)} newest novel views, {len(older)} older")
    sched = [torch.optim.lr_scheduler.ExponentialLR(runner.optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)),
             torch.optim.lr_scheduler.ChainedScheduler([
                 torch.optim.lr_scheduler.LinearLR(runner.post_processing_optimizers[0], start_factor=0.01, total_iters=1000),
                 torch.optim.lr_scheduler.ExponentialLR(runner.post_processing_optimizers[0], gamma=0.01 ** (1.0 / cfg.max_steps))])]
    for _ in range(start):
        for s in sched:
            s.step()
    loader = torch.utils.data.DataLoader(runner.trainset, batch_size=1, shuffle=True, num_workers=6,
                                         persistent_workers=True, pin_memory=True)
    it = iter(loader)
    stats = {"real": [], "novel": [], "mask": []}
    for step in range(start, start + steps):
        real = not (newest or older) or random.random() >= NOVEL_P
        if not real:
            c2w, K, target = random.choice(newest if (not older or random.random() < 0.7) else older)
            h, w = target.shape[:2]
            colors, _, _ = runner.rasterize_splats(c2w[None], K[None], w, h, sh_degree=cfg.sh_degree,
                                                   near_plane=cfg.near_plane, far_plane=cfg.far_plane)
            target = target.float() / 255
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
            image = data["image"][0]
            H, W = image.shape[:2]
            x0, y0 = random.randint(0, W - TW), random.randint(0, H - TH)
            c2w, ids = data["camtoworld"].to(dev), data["image_id"].to(dev)
            K = data["K"].to(dev).clone()
            K[:, 0, 2] -= x0
            K[:, 1, 2] -= y0
            pixels = image[y0:y0 + TH, x0:x0 + TW].to(dev)[None] / 255.0
            renders, alphas, info = runner.rasterize_splats(c2w, K, TW, TH, sh_degree=cfg.sh_degree,
                                                            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                                                            image_ids=ids, render_mode="RGB+ED")
            if densify:
                st.step_pre_backward(runner.splats, runner.optimizers, state, step, info)
            gy, gx = torch.meshgrid((y0 + torch.arange(TH, device=dev) + 0.5) / H,
                                    (x0 + torch.arange(TW, device=dev) + 0.5) / W, indexing="ij")
            colors = bilagrid_slice(runner.post_processing_module, torch.stack([gx, gy], -1)[None],
                                    renders[..., :3], ids[:, None])["rgb"]
            depths = renders[..., 3:4]
            m = 1 - F.max_pool2d(1 - (alphas.detach() > 0.5).float().permute(0, 3, 1, 2), ERODE, stride=1,
                                 padding=ERODE // 2).permute(0, 2, 3, 1)
            stats["mask"].append(m.mean().item())
            l1 =((colors - pixels).abs() * m).sum() / (3 * m.sum()).clamp(min=1)
            pred = colors * m + pixels * (1 - m)  # off the crop the tile scores itself, so no pull there
            loss = torch.lerp(l1, ssim_loss(pred.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)), cfg.ssim_lambda)
            pts, dep = data["points"][0] - torch.tensor([x0, y0]), data["depths"][0]
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < TW) & (pts[:, 1] >= 0) & (pts[:, 1] < TH)
            if inside.any():
                pts, dep = pts[inside].to(dev), dep[inside].to(dev)
                grid = torch.stack([pts[:, 0] / (TW - 1) * 2 - 1, pts[:, 1] / (TH - 1) * 2 - 1], -1)[None, :, None]
                d = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)[0, 0, :, 0]
                on = F.grid_sample(m.permute(0, 3, 1, 2), grid, align_corners=True)[0, 0, :, 0] > 0.5
                if on.any():
                    loss = loss + cfg.depth_lambda * depth_l1_loss(d[on][None], dep[on][None], scene_scale=runner.scene_scale)
            loss = loss + 10 * total_variation_loss(runner.post_processing_module.grids)
            stats["real"].append(l1.item())
        loss.backward()
        for opt in list(runner.optimizers.values()) + list(runner.post_processing_optimizers):
            opt.step()
            opt.zero_grad(set_to_none=True)
        for s in sched:
            s.step()
        if real and densify:
            st.step_post_backward(runner.splats, runner.optimizers, state, step, info, packed=cfg.packed,
                                  scene=getattr(runner, "scene", None))
            if len(runner.splats["means"]) > CAP and st.refine_stop_iter > step + 1:
                st.refine_stop_iter = step + 1
                print(f"  step {step + 1}: {len(runner.splats['means']):,} Gaussians, past the cap; densifying stops")
        if (step - start + 1) % 500 == 0:
            outside = int((~in_crop(runner)).sum())  # the crop's edge creeping outward shows here
            print(f"  step {step + 1}: real L1 {np.mean(stats['real'][-300:]):.4f}, "
                  f"novel L1 {np.mean(stats['novel'][-150:] or [0]):.4f}, {len(runner.splats['means']):,} Gaussians "
                  f"({outside:,} outside the crop), scored {np.mean(stats['mask'][-300:]):.0%} of the tile", flush=True)
        if (step - start + 1) % 3000 == 0 and step + 1 < start + steps:
            save(runner, step, out_ckpt[:-3] + "_part.pt")  # to look at mid-run; overwritten
    save(runner, start + steps - 1, out_ckpt)


def best_ref(P, eye, c2ws, K4, W4, H4):
    """The 4K photo that sees P best: in frame (5% margin), close, looking along the view's own direction."""
    vn = (P - eye) / np.linalg.norm(P - eye)
    best, score = 0, np.inf
    for j, c in enumerate(c2ws):
        x = (P - c[:3, 3]) @ c[:3, :3]
        if x[2] <= 0:
            continue
        u, v = K4[0, 0] * x[0] / x[2] + K4[0, 2], K4[1, 1] * x[1] / x[2] + K4[1, 2]
        if not (0.05 * W4 < u < 0.95 * W4 and 0.05 * H4 < v < 0.95 * H4):
            continue
        ray = P - c[:3, 3]
        dist = np.linalg.norm(ray)
        s = dist / max(float(vn @ ray) / dist, 0.2)
        if s < score:
            best, score = j, s
    return best


def render(clip, ckpt, out):
    use_4k(clip)
    for d in ("full_render", "alpha", "cam", "render", "ref"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    runner = make_runner(clip, 100000)
    load(runner, ckpt)
    splats = crop_splats(runner, in_crop(runner))
    middle, up, _, _, _, reach, heading = web_frame(runner.parser)
    ts, p = runner.trainset, runner.trainset.parser
    K2 = photo_K(ts)
    K2[:2] *= RW / dr.NW
    c2ws = [p.camtoworlds[i].astype(np.float64) for i in ts.indices]
    K4 = p.Ks_dict[p.camera_ids[ts.indices[0]]].astype(np.float64)
    W4, H4 = p.imsize_dict[p.camera_ids[ts.indices[0]]]
    aims = np.array([aim_of(c, middle, up) for c in c2ws])
    aims -= ((aims - middle) @ up)[:, None] * up
    centres = [aims[np.argmin(np.linalg.norm(aims - middle, axis=1))]]
    gap = np.linalg.norm(aims - centres[0], axis=1)
    while len(centres) < min(CENTRES, len(aims)):
        i = int(np.argmax(gap))
        centres.append(aims[i])
        gap = np.minimum(gap, np.linalg.norm(aims - aims[i], axis=1))
    views = []
    for ci, c in enumerate(centres):
        for el in ELEVATIONS:
            for side in SIDES:
                h = rot_about(up, side) @ -heading
                e = np.radians(el)
                eye = c + reach * BACK * (np.cos(e) * h + np.sin(e) * up)
                views.append((f"h_c{ci:02d}_e{el}_s{side:03d}", look_at(eye, c, up if el < 90 else -h)))
    photos, depths = {}, {}
    Kq = K4.copy()
    Kq[:2] /= 4
    u, v = np.meshgrid(np.arange(RW) + 0.5, np.arange(RH) + 0.5)
    rays = np.stack([(u - K2[0, 2]) / K2[0, 0], (v - K2[1, 2]) / K2[1, 1], np.ones_like(u)], -1)
    n_tiles = 0
    for name, c2w in views:
        img, alpha = draw(runner, splats, c2w, K2, RW, RH)
        z = draw(runner, splats, c2w, K2, RW, RH, mode="ED")[0][..., 0].cpu().numpy()
        img = (img.clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
        a = alpha.cpu().numpy()
        cv2.imwrite(f"{out}/full_render/{name}.png", img)
        cv2.imwrite(f"{out}/alpha/{name}.png", (a.clip(0, 1) * 255).round().astype(np.uint8))
        np.save(f"{out}/cam/{name}.npy", np.stack([c2w, np.pad(K2, ((0, 1), (0, 1)))]).astype(np.float32))
        world = (rays * z[..., None]) @ c2w[:3, :3].T + c2w[:3, 3]
        for ty in range(3):
            for tx in range(3):
                sl = (slice(ty * STRIDE_Y, ty * STRIDE_Y + TILE_H), slice(tx * STRIDE_X, tx * STRIDE_X + TILE_W))
                drawn = a[sl] > 0.5
                if drawn.mean() < 0.05:
                    continue
                X = world[sl]
                j = best_ref(np.median(X[drawn], axis=0), c2w[:3, 3], c2ws, K4, W4, H4)
                if j not in photos:
                    photos[j] = cv2.imread(p.image_paths[ts.indices[j]])
                    depths[j] = draw(runner, splats, c2ws[j], Kq, W4 // 4, H4 // 4, mode="ED")[0][..., 0].cpu().numpy()
                xr = (X - c2ws[j][:3, 3]) @ c2ws[j][:3, :3]
                zr = np.where(xr[..., 2] > 1e-6, xr[..., 2], 1e-6)
                ur, vr = K4[0, 0] * xr[..., 0] / zr + K4[0, 2], K4[1, 1] * xr[..., 1] / zr + K4[1, 2]
                ok = drawn & (xr[..., 2] > 1e-6) & (ur >= 0) & (ur < W4) & (vr >= 0) & (vr < H4)
                dq = depths[j]
                qi = np.clip((vr / 4).astype(int), 0, dq.shape[0] - 1)
                qj = np.clip((ur / 4).astype(int), 0, dq.shape[1] - 1)
                ok &= zr <= dq[qi, qj] * 1.15  # only real occluders (grass, kerb); 5% speckled rough asphalt
                ref = cv2.remap(photos[j], (ur - 0.5).astype(np.float32), (vr - 0.5).astype(np.float32),
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                ref[~ok] = img[sl][~ok]
                tile = f"{name}_t{ty}{tx}"
                cv2.imwrite(f"{out}/render/{tile}.png", img[sl])
                cv2.imwrite(f"{out}/ref/{tile}.png", ref)
                n_tiles += 1
        if len(photos) > 40:
            photos.clear()
            depths.clear()
    print(f"{len(views)} views at {RW}x{RH} looking down {ELEVATIONS} degrees, {n_tiles} tiles of {TILE_W}x{TILE_H} for Difix")


def stitch(out):
    os.makedirs(f"{out}/full", exist_ok=True)
    ramp_w = np.clip((np.arange(TILE_W) + 0.5) / FEATHER, 0, 1)
    ramp_h = np.clip((np.arange(TILE_H) + 0.5) / FEATHER, 0, 1)
    names = sorted(f[:-4] for f in os.listdir(f"{out}/cam") if f.endswith(".npy"))
    for name in names:
        base = cv2.imread(f"{out}/full_render/{name}.png").astype(np.float32)
        acc, wsum = np.zeros_like(base), np.zeros(base.shape[:2], np.float32)
        for ty in range(3):
            for tx in range(3):
                t = cv2.imread(f"{out}/difix_ref/{name}_t{ty}{tx}.png")
                if t is None:
                    continue
                wx, wy = np.ones(TILE_W), np.ones(TILE_H)
                if tx > 0:
                    wx = np.minimum(wx, ramp_w)
                if tx < 2:
                    wx = np.minimum(wx, ramp_w[::-1])
                if ty > 0:
                    wy = np.minimum(wy, ramp_h)
                if ty < 2:
                    wy = np.minimum(wy, ramp_h[::-1])
                w = np.outer(wy, wx).astype(np.float32)
                sl = (slice(ty * STRIDE_Y, ty * STRIDE_Y + TILE_H), slice(tx * STRIDE_X, tx * STRIDE_X + TILE_W))
                acc[sl] += t.astype(np.float32) * w[..., None]
                wsum[sl] += w
        wb = np.clip(1 - wsum, 0, 1)
        full = (acc + base * wb[..., None]) / (wsum + wb)[..., None]
        cv2.imwrite(f"{out}/full/{name}.png", full.round().clip(0, 255).astype(np.uint8))
    print(f"{len(names)} views stitched")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6] == "1", sys.argv[7:])
    elif mode == "render":
        render(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "stitch":
        stitch(sys.argv[2])
