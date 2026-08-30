"""
Extract SMPL-X parameters from SMPLest-X mesh OBJs via inverse fit.

For each `<smplestx_dir>/pred_smplx/{frame:04d}_{view:02d}.obj` (10475-vertex
SMPL-X mesh in world frame), recover (betas, body_pose, global_orient, trans)
by minimizing per-vertex L2 to the reference mesh. Output npz files in
`<smplestx_dir>/init_smplxinit/{frame:04d}_{view:02d}.npz`.

These params are intended as init for the multi-view fitting pipeline (EXP-6).
We only consume `betas` + `body_pose` downstream; `global_orient` / `trans` are
saved for completeness but the fitting still re-finds them in its own frame
(SMPLest-X output is in dataset world frame, fitting works in proxy frame
with its own scale).

Usage:
    python extract_smplx_init.py \
        --smplestx_dir /path/to/evaluate/<dataset>/smplestx \
        --gender neutral --device cuda:0 --iters 300
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import torch

import smplx


_OBJ_NAME_RE = re.compile(r"^(?P<sid>\d{4})_(?P<vid>\d{2})\.obj$")


def load_obj_verts(path: str) -> np.ndarray:
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float32)


def inverse_fit_smplx(target_verts: torch.Tensor, smplx_model, device,
                      iters: int = 200, lr: float = 0.05,
                      lr_decay_at: float = 0.6, lr_decay_factor: float = 0.2,
                      verbose: bool = False):
    """Recover (betas, body_pose, global_orient, trans, +hand/jaw poses) from
    a SMPL-X mesh by per-vertex L2 inverse fit.
    target_verts: (V, 3) tensor on `device`. V must be 10475 (SMPL-X topology).
    """
    V = target_verts.shape[0]
    assert V == 10475, f"expected 10475 SMPL-X verts, got {V}"

    betas = torch.zeros(10, device=device, requires_grad=True)
    body_pose = torch.zeros(63, device=device, requires_grad=True)
    global_orient = torch.zeros(3, device=device, requires_grad=True)
    left_hand_pose = torch.zeros(45, device=device, requires_grad=True)
    right_hand_pose = torch.zeros(45, device=device, requires_grad=True)
    jaw_pose = torch.zeros(3, device=device, requires_grad=True)
    trans = target_verts.mean(dim=0).clone().detach().requires_grad_(True)

    opt = torch.optim.Adam(
        [betas, body_pose, global_orient, left_hand_pose, right_hand_pose, jaw_pose, trans],
        lr=lr,
    )

    decay_iter = int(iters * lr_decay_at)
    last_loss = None
    for it in range(iters):
        if it == decay_iter:
            for g in opt.param_groups:
                g["lr"] *= lr_decay_factor
        out = smplx_model(
            betas=betas.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            left_hand_pose=left_hand_pose.unsqueeze(0),
            right_hand_pose=right_hand_pose.unsqueeze(0),
            jaw_pose=jaw_pose.unsqueeze(0),
            transl=trans.unsqueeze(0),
        )
        v_pred = out.vertices[0]
        loss = (v_pred - target_verts).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (it % 50 == 0 or it == iters - 1):
            print(f"  iter {it:3d}: vert-MSE = {loss.item():.6f} m^2")
        last_loss = loss.item()

    return {
        "betas": betas.detach().cpu().numpy().astype(np.float32),
        "body_pose": body_pose.detach().cpu().numpy().astype(np.float32),
        "global_orient": global_orient.detach().cpu().numpy().astype(np.float32),
        "left_hand_pose": left_hand_pose.detach().cpu().numpy().astype(np.float32),
        "right_hand_pose": right_hand_pose.detach().cpu().numpy().astype(np.float32),
        "jaw_pose": jaw_pose.detach().cpu().numpy().astype(np.float32),
        "trans": trans.detach().cpu().numpy().astype(np.float32),
        "vert_mse_final": np.float32(last_loss if last_loss is not None else float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplestx_dir", required=True,
                    help="path to <evaluate>/<dataset>/smplestx (contains pred_smplx/)")
    ap.add_argument("--out_subdir", default="init_smplxinit")
    ap.add_argument("--model_folder",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                    help="folder containing smplx/SMPLX_*.npz body models (default: ./data)")
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--views", type=str, default="all",
                    help="comma-separated view ids to process, or 'all'")
    ap.add_argument("--limit", type=int, default=-1, help="max OBJs to process; -1 = all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = os.path.join(args.smplestx_dir, "pred_smplx")
    out_dir = os.path.join(args.smplestx_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    objs_all = sorted(glob.glob(os.path.join(src_dir, "*.obj")))
    if args.views == "all":
        objs = objs_all
    else:
        view_set = set(int(v) for v in args.views.split(","))
        objs = []
        for p in objs_all:
            m = _OBJ_NAME_RE.match(os.path.basename(p))
            if m and int(m.group("vid")) in view_set:
                objs.append(p)
    if args.limit > 0:
        objs = objs[: args.limit]
    print(f"[InvFit] {src_dir} → {out_dir}")
    print(f"[InvFit] found {len(objs)} obj files; iters={args.iters} lr={args.lr}")

    device = torch.device(args.device)
    smplx_model = smplx.create(
        args.model_folder, model_type="smplx", gender=args.gender,
        ext="npz", num_betas=10, flat_hand_mean=False, use_pca=False,
    ).to(device)
    smplx_model.eval()

    t0 = time.perf_counter()
    n_done, n_skip = 0, 0
    mse_list = []
    for i, obj_path in enumerate(objs):
        name = os.path.basename(obj_path)
        m = _OBJ_NAME_RE.match(name)
        if m is None:
            print(f"[Skip] non-matching name: {name}")
            continue
        out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue

        try:
            verts_np = load_obj_verts(obj_path)
            if verts_np.shape[0] != 10475:
                print(f"[Skip] {name}: {verts_np.shape[0]} verts != 10475")
                continue
            target = torch.from_numpy(verts_np).to(device)
            params = inverse_fit_smplx(target, smplx_model, device,
                                       iters=args.iters, lr=args.lr)
            np.savez(out_path,
                     betas=params["betas"],
                     body_pose=params["body_pose"],
                     global_orient=params["global_orient"],
                     left_hand_pose=params["left_hand_pose"],
                     right_hand_pose=params["right_hand_pose"],
                     jaw_pose=params["jaw_pose"],
                     trans=params["trans"],
                     vert_mse_final=params["vert_mse_final"],
                     source_obj=name)
            n_done += 1
            mse_list.append(float(params["vert_mse_final"]))
            if (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - t0
                rate = (n_done + n_skip) / max(elapsed, 1e-6)
                print(f"  [{i + 1}/{len(objs)}] done={n_done} skip={n_skip} "
                      f"mse_last={params['vert_mse_final']:.2e} m^2 "
                      f"({rate:.2f}/s)")
        except Exception as e:
            print(f"[Err] {name}: {type(e).__name__}: {e}")

    elapsed = time.perf_counter() - t0
    if mse_list:
        mse_arr = np.asarray(mse_list)
        print(f"[InvFit] done={n_done} skip={n_skip} elapsed={elapsed:.1f}s "
              f"vert-MSE mean={mse_arr.mean():.2e} median={np.median(mse_arr):.2e} "
              f"max={mse_arr.max():.2e} m^2")
    else:
        print(f"[InvFit] done={n_done} skip={n_skip} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
