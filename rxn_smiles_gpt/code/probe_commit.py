"""Verify how /vol persistence behaves in-container: write, commit, re-read."""
import os

import modal

app = modal.App("uspto50k-probe")
vol = modal.Volume.from_name("uspto50k", create_if_missing=True)


image = modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.6")


@app.function(image=image, gpu="L4", cpu=1, timeout=60 * 3, volumes={"/vol": vol})
def probe():
    import sys
    sys.path.insert(0, "/app/code")
    os.makedirs("/vol/runs/probe", exist_ok=True)
    with open("/vol/runs/probe/hello.txt", "w") as f:
        f.write("persist-test")
    print("wrote", flush=True)
    print("pre-commit exists:", os.path.exists("/vol/runs/probe/hello.txt"), flush=True)
    r = vol.commit()
    print("commit returned:", r, flush=True)
    print("post-commit exists:", os.path.exists("/vol/runs/probe/hello.txt"), flush=True)
    print("pre-commit second file:", flush=True)
    with open("/vol/runs/probe/no_commit.txt", "w") as f:
        f.write("written-after-commit")
    return {"done": True}


@app.local_entrypoint()
def main():
    probe.remote()