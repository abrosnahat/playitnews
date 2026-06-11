"""
PlayItNews persistent MuseTalk worker.

Runs inside the isolated .venv-musetalk (Python 3.10, torch cu128) with the
working directory set to the MuseTalk repo root so the upstream relative model
paths (./models/...) resolve.

Design
------
The upstream scripts/realtime_inference.py does all of its model loading and
avatar orchestration inside `if __name__ == "__main__":`, so it cannot be
imported and reused directly. Instead we import it as a module and *inject* the
globals its `Avatar` class expects (vae/unet/pe/fp/whisper/audio_processor/...).
Python resolves a function's free names in its own module's globals, so setting
`ri.vae = ...` makes `Avatar.inference` see it.

The avatar (face crop coords, latents, blending masks) is prepared ONCE and
cached on disk; every subsequent job only runs the UNet + VAE decode, which is
the fast real-time path. Jobs arrive as one JSON object per stdin line:

    {"audio_path": "C:/.../voice.wav", "out_path": "C:/.../head.mp4"}

For each job the worker prints exactly one line:

    RESULT <out_path>     on success
    ERROR <message>       on failure

and prints `READY` once after models are loaded and the avatar is prepared.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
import types

# Upstream preprocessing prints status with CJK characters (e.g. 「」). On a
# Windows console using a legacy code page (cp1251/cp866) that raises
# UnicodeEncodeError and kills avatar preparation. Force UTF-8 on our streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Triton has no Windows wheel, so torch.compile / TorchDynamo cannot generate
# GPU kernels here. face-alignment 1.5 calls torch.compile on its landmark net;
# disable Dynamo entirely so everything runs in eager mode instead of crashing
# with TritonMissing.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from transformers import WhisperModel

# PyTorch 2.6 flipped torch.load's default to weights_only=True, which breaks
# loading MuseTalk's legacy .tar / pickled checkpoints (resnet18, sfd detector,
# sd-vae). These weights ship with the trusted model bundle, so force the legacy
# behaviour for the whole worker process.
_orig_torch_load = torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


torch.load = _torch_load_compat


def _build_args(ns_overrides: dict) -> argparse.Namespace:
    """Recreate the argparse.Namespace that realtime_inference's globals expect."""
    args = argparse.Namespace(
        version="v15",
        ffmpeg_path="",  # ffmpeg already on PATH
        gpu_id=0,
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        unet_model_path="./models/musetalkV15/unet.pth",
        whisper_dir="./models/whisper",
        bbox_shift=0,
        result_dir="./results",
        extra_margin=10,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        batch_size=20,
        output_vid_name=None,
        use_saved_coord=False,
        saved_coord=False,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        skip_save_images=False,
    )
    for k, v in ns_overrides.items():
        setattr(args, k, v)
    return args


def _avatar_id_for(video_path: str) -> str:
    h = hashlib.md5(os.path.abspath(video_path).encode("utf-8")).hexdigest()[:10]
    return f"playit_{h}"


def _source_fingerprint(video_path: str) -> str:
    """A cheap content signature for the source video (size + mtime)."""
    try:
        st = os.stat(video_path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return ""


def _fingerprint_path(avatar_dir: str) -> str:
    return os.path.join(avatar_dir, "source.fingerprint")


def _read_fingerprint(avatar_dir: str) -> str:
    try:
        with open(_fingerprint_path(avatar_dir), "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _write_fingerprint(avatar_dir: str, fingerprint: str) -> None:
    try:
        os.makedirs(avatar_dir, exist_ok=True)
        with open(_fingerprint_path(avatar_dir), "w", encoding="utf-8") as f:
            f.write(fingerprint)
    except OSError:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--avatar_video", required=True,
                   help="Path to the source talking-person loop video.")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--extra_margin", type=int, default=10)
    p.add_argument("--parsing_mode", default="jaw")
    cli = p.parse_args()
    # Import the upstream module (its heavy work is guarded by __main__).
    import scripts.realtime_inference as ri
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing

    args = _build_args({
        "batch_size": cli.batch_size,
        "fps": cli.fps,
        "extra_margin": cli.extra_margin,
        "parsing_mode": cli.parsing_mode,
    })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    # Inject the globals Avatar.* reads from its own module namespace.
    ri.args = args
    ri.device = device
    ri.vae = vae
    ri.unet = unet
    ri.pe = pe
    ri.timesteps = timesteps
    ri.audio_processor = audio_processor
    ri.weight_dtype = weight_dtype
    ri.whisper = whisper
    ri.fp = fp

    # Prepare the avatar once (cached on disk between runs).
    avatar_id = _avatar_id_for(cli.avatar_video)
    avatar_dir = f"./results/{args.version}/avatars/{avatar_id}"
    # Treat the avatar as cached only if the prep actually finished, i.e. the
    # latents file is present. A bare directory from a crashed prep run must be
    # discarded so Avatar re-prepares instead of failing on a missing latents.pt.
    #
    # The avatar_id is derived from the source video PATH, so replacing the file
    # in place (same path) would otherwise reuse a stale cache. Guard against
    # that by also fingerprinting the source video (size + mtime): if it changed
    # since the cache was built, drop the cache so the avatar re-prepares.
    fingerprint = _source_fingerprint(cli.avatar_video)
    cached = (
        os.path.exists(os.path.join(avatar_dir, "latents.pt"))
        and _read_fingerprint(avatar_dir) == fingerprint
    )
    if os.path.exists(avatar_dir) and not cached:
        shutil.rmtree(avatar_dir, ignore_errors=True)
    avatar = ri.Avatar(
        avatar_id=avatar_id,
        video_path=cli.avatar_video,
        bbox_shift=0,
        batch_size=args.batch_size,
        preparation=not cached,
    )
    if not cached:
        _write_fingerprint(avatar_dir, fingerprint)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    job_n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            audio_path = job["audio_path"]
            out_path = job["out_path"]
            job_n += 1
            out_name = f"job_{job_n:05d}"
            avatar.inference(audio_path, out_name, args.fps, skip_save_images=False)
            produced = os.path.join(
                f"./results/{args.version}/avatars/{avatar_id}/vid_output",
                out_name + ".mp4",
            )
            if not os.path.exists(produced):
                raise RuntimeError(f"MuseTalk produced no output at {produced}")
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            shutil.copy2(produced, out_path)
            try:
                os.remove(produced)
            except OSError:
                pass
            sys.stdout.write(f"RESULT {out_path}\n")
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 — report, keep serving
            sys.stdout.write(f"ERROR {exc!r}\n")
            sys.stdout.flush()
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()


if __name__ == "__main__":
    main()
