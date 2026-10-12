"""
Modal app for the USPTO-50k retro GPT. L4 GPU, hard timeouts (budget-capped).

Usage:
    modal run code/modal_app.py --tiny            # timing probe on L4
    modal run code/modal_app.py --epochs 14 --beam 10
"""

import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("uspto50k-retro")
vol = modal.Volume.from_name("uspto50k", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.6", "numpy", "pandas", "pyarrow", "rdkit")
         .add_local_dir(os.path.join(ROOT, "code"), "/app/code"))


@app.function(image=image, gpu="L4", cpu=4, memory=8192, timeout=60 * 12,
              volumes={"/vol": vol})
def tiny_test(steps: int = 40):
    """Timing probe on the real config (synthetic tokens, no data needed)."""
    import sys
    import time

    sys.path.insert(0, "/app/code")
    import torch
    import torch.nn.functional as F

    import gpu_core

    model = gpu_core.make_model(device="cuda")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    V = gpu_core.vocab_size()
    xb = torch.randint(0, V, (512, 255), device="cuda")
    yb = torch.randint(0, V, (512, 255), device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    ts = []
    for i in range(steps):
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    t_med = sorted(ts)[len(ts) // 2]
    steps_per_epoch = -(-48995 // 512)  # ceil
    s_epoch = t_med * steps_per_epoch
    print(f"median step {t_med*1000:.0f} ms -> full epoch ~= {s_epoch/60:.1f} min",
          flush=True)
    print(f"tokens/s {512*255/t_med:,.0f}", flush=True)
    return {"step_ms": t_med * 1000, "epoch_min": s_epoch / 60}


@app.function(image=image, gpu="L4", cpu=2, memory=8192, timeout=60 * 8,
              volumes={"/vol": vol})
def tiny_end2end():
    """Miniature of the whole job on real data — the pre-flight the two
    erased training runs never had. Verifies, in order: the resolved
    checkpoint path is on the volume, save+commit lands a readable file,
    the beam decoder produces SMILES on real val rows."""
    import sys
    sys.path.insert(0, "/app/code")
    import torch
    import gpu_core

    gpu_core.KEEP_CKPT = vol.commit
    print(f"ckpt path: {gpu_core.CKPT}", flush=True)
    assert gpu_core.RUNS.startswith("/vol"), \
        f"checkpoint path bypasses the volume mount: {gpu_core.RUNS}"

    # save a real (untrained) checkpoint of the real config and commit
    model = gpu_core.make_model(device="cuda")
    gpu_core.save_ckpt(model)
    torch.cuda.synchronize()
    size = os.path.getsize("/vol/runs/retro_gpt.pt")
    print(f"saved {size/1e6:.0f} MB, committed to volume", flush=True)

    # read it back from the mount — not just from container pages
    d = torch.load("/vol/runs/retro_gpt.pt", map_location="cpu", weights_only=True)
    n = sum(v.numel() for v in d["state_dict"].values())
    print(f"read back {n:,} params, config {d['config']}", flush=True)

    # decode 4 real val rows through the beam path
    res = gpu_core.beam_eval(model, beam=5, eval_rows=4, device="cuda")
    print("val-gt first four rows:", flush=True)
    import json
    with open("/vol/data/retro_val_gt.json") as f:
        gt = json.load(f)
    for i in range(4):
        print(f"  [{i}] pred-vs-gt: {gt['gts_react_canon'][i]}", flush=True)
    print(f"tiny_end2end OK, beam metrics on 4 rows: {res}", flush=True)


@app.function(image=image, gpu="L4", cpu=4, memory=8192, timeout=60 * 95,
              volumes={"/vol": vol})
def train(epochs: int = 30, beam: int = 10, eval_rows: int = 500):
    import sys
    sys.path.insert(0, "/app/code")
    import gpu_core
    # persist every checkpoint save; function-side writes to the mount do
    # not survive container exit without an explicit commit
    gpu_core.KEEP_CKPT = vol.commit
    res = gpu_core.run(epochs, beam=beam, eval_rows=eval_rows)
    vol.commit()
    return res


@app.local_entrypoint()
def main(tiny: bool = False, end2end: bool = False, epochs: int = 30,
         beam: int = 10, eval_rows: int = 500):
    if tiny:
        tiny_test.remote(steps=40)
    elif end2end:
        tiny_end2end.remote()
    else:
        train.remote(epochs=epochs, beam=beam, eval_rows=eval_rows)