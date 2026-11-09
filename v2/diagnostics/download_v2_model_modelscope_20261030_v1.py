"""Download the exact V2 model bytes through ModelScope; no package installation."""
import argparse, hashlib, json, os, re, shutil, sys, time, uuid
from pathlib import Path
import requests
from tqdm import tqdm

MANIFEST = {"model_id":"Qwen/Qwen2.5-1.5B-Instruct","requested_revision":"989aa79","resolved_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306","files":{"config.json":{"size":660,"sha256":"98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670","modelscope_revision":"a3c2dc17129625b1e51caf21ab486d32d1f12982","hf_git_blob_sha1":"f81ead14ab072d65a07817f83a3ee0e5a1890d10"},"generation_config.json":{"size":242,"sha256":"e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6","modelscope_revision":"a3c2dc17129625b1e51caf21ab486d32d1f12982","hf_git_blob_sha1":"dfc11073787daf1b0f9c0f1499487ab5f4c93738"},"merges.txt":{"size":1671839,"sha256":"599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3","modelscope_revision":"a3c2dc17129625b1e51caf21ab486d32d1f12982","hf_git_blob_sha1":"20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"},"model.safetensors":{"size":3087467144,"sha256":"dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee","modelscope_revision":"a3c2dc17129625b1e51caf21ab486d32d1f12982"},"tokenizer.json":{"size":7031645,"sha256":"c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539","modelscope_revision":"a3c2dc17129625b1e51caf21ab486d32d1f12982","hf_git_blob_sha1":"443909a61d429dff23010e5bddd28ff530edda00"},"tokenizer_config.json":{"size":7305,"sha256":"5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583","modelscope_revision":"bc68ae58812462d8bf5d9ba5131317ed7fb71665","hf_git_blob_sha1":"07bfe0640cb5a0037f9322287fbfc682806cf672"},"vocab.json":{"size":2776833,"sha256":"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910","modelscope_revision":"a3c2dc17129625b1e51caf21ab486d32d1f12982","hf_git_blob_sha1":"4783fe10ac3adce15ac8f358ef5462739852c569"}}}
ENDPOINT = "https://www.modelscope.cn/api/v1/models/" + MANIFEST["model_id"] + "/repo"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()

def verified(path, spec):
    return path.is_file() and path.stat().st_size == spec["size"] and digest(path) == spec["sha256"]

def params(name):
    return {"Revision": MANIFEST["files"][name]["modelscope_revision"], "FilePath": name}

def transient(exc):
    if isinstance(exc, (requests.ConnectionError, requests.Timeout,
                        requests.exceptions.ChunkedEncodingError)):
        return True
    return getattr(getattr(exc, "response", None), "status_code", None) in (429, 500, 502, 503, 504)

def download_file(session, name, spec, part):
    """Resume only at a server-confirmed range; verify every final byte."""
    part.parent.mkdir(parents=True, exist_ok=True)
    size = spec["size"]
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            raise ValueError("Partial file exceeds pinned size: " + name)
        if offset == size:
            if not verified(part, spec):
                raise ValueError("Pinned SHA-256 differs: " + name)
            return
        print(f"Fetching {name}: {offset:,}/{size:,} bytes", flush=True)
        try:
            headers = {"Range": f"bytes={offset}-{size-1}", "Accept-Encoding": "identity"}
            with session.get(ENDPOINT, params=params(name), headers=headers,
                             stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                mode = "ab" if offset else "wb"
                if response.status_code == 206:
                    expected = f"bytes {offset}-{size-1}/{size}"
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("Unexpected Content-Range: " + name)
                elif response.status_code == 200:
                    if offset:
                        backup = part.with_name(part.name + ".preserved-" + uuid.uuid4().hex[:8])
                        part.rename(backup)
                        print("Server ignored Range; preserved partial and restarting file.", flush=True)
                    offset, mode = 0, "wb"
                else:
                    raise ValueError("Unexpected download status: " + str(response.status_code))
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != size-offset:
                    raise ValueError("Response length differs from pinned size: " + name)
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Received an HTML page instead of a model file")
                with part.open(mode) as f, tqdm(total=size, initial=offset, unit="B",
                        unit_scale=True, desc=name, mininterval=1) as progress:
                    written = offset
                    for chunk in response.iter_content(chunk_size=2**20):
                        if not chunk:
                            continue
                        if written + len(chunk) > size:
                            raise ValueError("Download exceeds pinned size")
                        f.write(chunk)
                        written += len(chunk)
                        progress.update(len(chunk))
                if written != size:
                    raise requests.exceptions.ChunkedEncodingError("Incomplete download: " + name)
            if not verified(part, spec):
                raise ValueError("Pinned SHA-256 differs: " + name)
            print("Verified SHA-256: " + name, flush=True)
            return
        except requests.RequestException as exc:
            if not transient(exc) or attempt == 3:
                raise
            delay = (5, 20, 60)[attempt]
            print(f"{type(exc).__name__}; retrying in {delay}s (partial retained).", flush=True)
            time.sleep(delay)

def seed_partial(session, cache, part, spec):
    """Copy an existing HF partial after checking its prefix against ModelScope."""
    if part.exists():
        return
    candidates = [p for p in (cache/".cache/huggingface/download").rglob("*.incomplete")
                  if "model.safetensors" in p.name and 0 < p.stat().st_size < spec["size"]]
    if not candidates:
        return
    candidate = max(candidates, key=lambda p: p.stat().st_size)
    width = min(65536, candidate.stat().st_size)
    print(f"Checking existing HF partial ({candidate.stat().st_size:,} bytes).", flush=True)
    with session.get(ENDPOINT, params=params("model.safetensors"),
                     headers={"Range": f"bytes=0-{width-1}", "Accept-Encoding": "identity"},
                     stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes 0-{width-1}/{spec['size']}":
            print("Prefix range unavailable; original partial retained, new download starts at zero.", flush=True)
            return
        prefix = response.raw.read(width)
    with candidate.open("rb") as f:
        if len(prefix) != width or f.read(width) != prefix:
            raise ValueError("Existing HF partial prefix differs from pinned model")
    part.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, part)
    print("Copied existing partial for resume; original HF partial retained.", flush=True)

def publish(cache, prepared):
    """Publish only a fully validated model; refuse to replace existing files."""
    for name, spec in MANIFEST["files"].items():
        target = cache/name
        if target.exists() and not verified(target, spec):
            raise ValueError("Existing cache file differs; preserved without replacement: " + name)
        if not target.exists() and not verified(prepared/name, spec):
            raise ValueError("Staging file is not verified: " + name)
    for name in MANIFEST["files"]:
        target = cache/name
        if not target.exists():
            # Atomic create without replacement, on the same filesystem.
            os.link(prepared/name, target)
    identity = {key: MANIFEST[key] for key in ("model_id", "requested_revision", "resolved_revision")}
    identity["files_sha256"] = {name: spec["sha256"] for name, spec in MANIFEST["files"].items()}
    path = cache/"model_identity.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf8"))
        for key in ("model_id", "resolved_revision"):
            if saved[key] != identity[key]:
                raise ValueError("Existing model identity differs")
        if saved["files_sha256"] != identity["files_sha256"]:
            raise ValueError("Existing model manifest differs")
    else:
        temporary = prepared/("identity-" + uuid.uuid4().hex + ".json")
        temporary.write_text(json.dumps(identity, indent=2)+"\n", encoding="utf8")
        os.link(temporary, path)
    return identity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/rivermind-data/AgenticPostTrainingLab/v2"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root/"configs/baseline.json").read_text(encoding="utf8"))
    if config["model_id"] != MANIFEST["model_id"] or config["revision"] != MANIFEST["requested_revision"]:
        raise ValueError("Project model pin differs")
    cache = root/".cache/qwen2.5-1.5b-instruct"
    cache.mkdir(parents=True, exist_ok=True)
    prepared = cache/".modelscope-pinned-v1"
    prepared.mkdir(exist_ok=True)
    receipt_folder = root/"artifacts"/("modelscope-import-20261030-"+uuid.uuid4().hex[:8])
    receipt_folder.mkdir(parents=True, exist_ok=False)
    receipt = {"manifest": MANIFEST, "download_endpoint": ENDPOINT, "passed": False,
               "script_sha256": digest(__file__)}
    started = time.monotonic()
    print("ModelScope download started. Pinned HF revision: " + MANIFEST["resolved_revision"], flush=True)
    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "apt-v2-pinned-download/1", "Accept-Encoding": "identity"})
            for name, spec in MANIFEST["files"].items():
                target, part = cache/name, prepared/name
                if target.exists():
                    if not verified(target, spec):
                        raise ValueError("Existing cache file differs: " + name)
                    print("Reusing verified cache: " + name, flush=True)
                    continue
                if name == "model.safetensors":
                    seed_partial(session, cache, part, spec)
                download_file(session, name, spec, part)
        identity = publish(cache, prepared)
        receipt.update(passed=True, identity=identity)
        print("Model cache ready. Run: python diagnostics/check_environment.py", flush=True)
        return 0
    except Exception as exc:
        receipt.update(error_type=type(exc).__name__)
        print("Download failed: " + type(exc).__name__ + "; partial files retained.", flush=True)
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic()-started
        (receipt_folder/"download_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
        print("Receipt: " + str(receipt_folder/"download_receipt.json"), flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
