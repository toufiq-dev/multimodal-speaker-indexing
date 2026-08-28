"""Process-level runtime hardening: ABI locks, accelerator policy, VRAM lifecycle.

This module is the SINGLE source of truth for everything that must be true
about the process before any model is constructed. It is imported for its
side effects by ``config``, which every engine imports first, so the NumPy
ABI check and the cuDNN search path are established before ``insightface``,
``onnxruntime`` or ``ctranslate2`` are ever touched.

Three previously-duplicated concerns live here now:

1. NumPy compatibility. The ``np.NaN``/``np.Inf`` aliases were pasted into
   ``engines/vision.py``, ``engines/vision_yolo.py`` and ``kaggle_setup.py``.
   Those aliases only paper over *pure-Python* references removed in NumPy
   2.0; the real breakage under NumPy 2.x is the C-extension ABI
   (``_ARRAY_API not found``), which no Python-level alias can repair. So the
   aliases are kept for the narrow case they do cover, and backed by a hard
   version assertion that fails loudly instead of at a random call site.

2. ONNX Runtime provider policy. ``providers=['CUDAExecutionProvider', ...]``
   is a *request*, not a guarantee: if the CUDA EP is unavailable (commonly
   because a CPU ``onnxruntime`` wheel has shadowed ``onnxruntime-gpu``) ORT
   silently falls back to CPU. ``assert_cuda_execution_provider`` turns that
   into an exception, and ``onnx_providers`` caps the ORT arena so it cannot
   starve the torch caching allocator on a shared device.

3. VRAM lifecycle. ``torch.cuda.empty_cache()`` releases only *torch's*
   cached blocks, and only for tensors that are already unreferenced. Calling
   it while a model is still in scope frees nothing, and CTranslate2 / ONNX
   Runtime allocations are outside torch's allocator entirely.
   ``with_model()`` scopes a model so the reference is provably dropped
   before the cache is released.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Callable, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# 1. NumPy ABI lock
# --------------------------------------------------------------------------

#: insightface 0.7.3 and onnxruntime-gpu 1.19.2 ship wheels compiled against
#: the NumPy 1.x C ABI. Any 2.x runtime breaks them at import, not at use.
NUMPY_ABI_LOCK = "1.26"

_REMOVED_ALIASES = {"NaN": np.nan, "Inf": np.inf, "PINF": np.inf, "NINF": -np.inf}


def apply_numpy_compat() -> str:
    """Restore the NumPy 2.0-removed aliases insightface still references.

    Cheap and side-effect free, so it runs at ``config`` import for every
    entry point. It deliberately does NOT assert the ABI: the audio, fusion
    and evaluation paths are pure Python/NumPy and run fine on NumPy 2.x, and
    failing them would make the test suite unrunnable off-target. The hard
    check lives in :func:`assert_numpy_abi`, called where it matters.
    """
    for alias, value in _REMOVED_ALIASES.items():
        if not hasattr(np, alias):
            setattr(np, alias, value)
    return np.__version__


def assert_numpy_abi() -> None:
    """Fail before importing a wheel compiled against the NumPy 1.x C ABI.

    Call this immediately *before* importing insightface / onnxruntime. Under
    NumPy 2.x those extensions abort with ``_ARRAY_API not found`` or
    ``numpy.core.multiarray failed to import``, which no Python-level alias
    can repair — the aliases in :func:`apply_numpy_compat` cover only pure
    Python references and are a false safety net for this failure mode.

    Set ``ALLOW_NUMPY_ABI_DRIFT=1`` to bypass on a dev machine whose wheels
    were built for the local NumPy.
    """
    if os.getenv("ALLOW_NUMPY_ABI_DRIFT", "") == "1":
        return
    if not np.__version__.startswith(NUMPY_ABI_LOCK):
        raise RuntimeError(
            f"NumPy ABI lock broken: found {np.__version__}, need "
            f"{NUMPY_ABI_LOCK}.x. insightface==0.7.3 and onnxruntime-gpu==1.19.2 "
            f"are compiled against the NumPy 1.x C ABI and will fail at import "
            f"with '_ARRAY_API not found'. Reinstall with "
            f"`pip install -c constraints.txt -r requirements.txt`, or set "
            f"ALLOW_NUMPY_ABI_DRIFT=1 to run the non-vision stages anyway."
        )


# --------------------------------------------------------------------------
# 2. cuDNN discovery for CTranslate2
# --------------------------------------------------------------------------

def cudnn_library_dir() -> Optional[str]:
    """Locate the cuDNN shared objects shipped inside the torch wheel."""
    try:
        import nvidia.cudnn  # type: ignore
    except Exception:
        return None
    lib = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
    return lib if os.path.isdir(lib) else None


def ensure_cudnn_on_path() -> Optional[str]:
    """Prepend the wheel-local cuDNN directory to ``LD_LIBRARY_PATH``.

    CTranslate2 >= 4.5 links cuDNN 9 and resolves it through the dynamic
    loader, but ``torch==2.5.1+cu121`` installs cuDNN 9.1 under
    ``site-packages/nvidia/cudnn/lib`` which is not on the default search
    path. Without this, ``WhisperModel(device="cuda")`` dies with
    ``Could not load library libcudnn_ops_infer.so``.

    Must run before CTranslate2 is imported to affect the current process on
    some loaders; it is always correct for child processes.
    """
    lib = cudnn_library_dir()
    if not lib:
        return None
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (lib, current) if p)
    return lib


# --------------------------------------------------------------------------
# 3. ONNX Runtime provider policy
# --------------------------------------------------------------------------

#: Ceiling for the ORT CUDA arena. ORT and torch each hoard device memory;
#: on a 15 GB T4 running diarization/ASR alongside face analysis, an
#: unbounded arena is the difference between coexistence and OOM.
DEFAULT_ORT_GPU_MEM_LIMIT_GB = 4.0


def available_onnx_providers() -> List[str]:
    """Providers ORT can actually construct, or [] if ORT is missing."""
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def assert_cuda_execution_provider() -> None:
    """Fail loudly when CUDA was requested but ORT would silently use CPU."""
    providers = available_onnx_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            f"CUDA requested but onnxruntime only offers {providers or ['<onnxruntime missing>']}. "
            f"A CPU `onnxruntime` wheel (usually pulled in as a chromadb dependency) "
            f"shares onnxruntime-gpu's import path and shadows it. Fix with:\n"
            f"  pip uninstall -y onnxruntime && "
            f"pip install --force-reinstall --no-deps onnxruntime-gpu==1.19.2"
        )


def onnx_providers(use_cuda: bool, gpu_mem_limit_gb: Optional[float] = None) -> list:
    """Build an ORT provider list with an explicit, bounded CUDA arena.

    Args:
        use_cuda: Whether the CUDA EP should be requested at all.
        gpu_mem_limit_gb: Arena ceiling; defaults to
            ``ORT_GPU_MEM_LIMIT_GB`` or :data:`DEFAULT_ORT_GPU_MEM_LIMIT_GB`.
    """
    if not use_cuda:
        return ["CPUExecutionProvider"]

    if gpu_mem_limit_gb is None:
        gpu_mem_limit_gb = float(
            os.getenv("ORT_GPU_MEM_LIMIT_GB", DEFAULT_ORT_GPU_MEM_LIMIT_GB))

    return [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            # kNextPowerOfTwo (the default) over-allocates aggressively when
            # torch is also holding device memory.
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": int(gpu_mem_limit_gb * 1024 ** 3),
        }),
        "CPUExecutionProvider",
    ]


# --------------------------------------------------------------------------
# 4. VRAM lifecycle
# --------------------------------------------------------------------------

def release_gpu_memory() -> None:
    """Collect Python garbage, then return freed blocks to the driver.

    Order matters: ``empty_cache()`` only releases blocks with no live
    reference, so the collection must happen first.
    """
    gc.collect()
    try:
        import torch  # local import: keeps this module importable without torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def with_model(factory: Callable[[], Any],
               use: Callable[[Any], Any],
               label: str = "model") -> Any:
    """Load a model, use it, then provably free it before returning.

    Deliberately NOT a context manager. ``with loaded(...) as model:`` leaves
    ``model`` bound in the *caller's* frame after the block exits, so the
    release would run with a live reference and reclaim nothing — the very
    bug this helper exists to prevent. Passing a callable keeps the only
    strong reference inside this function, where ``del`` is authoritative.

    This matters because CTranslate2 and ONNX Runtime allocate outside torch's
    caching allocator: their device memory is returned when the Python object
    is collected, and never by ``empty_cache()`` alone.

    Example:
        >>> words, _ = with_model(                       # doctest: +SKIP
        ...     _load_model,
        ...     lambda m: transcribe_audio(path, model=m),
        ...     "faster-whisper")
    """
    model = factory()
    try:
        return use(model)
    finally:
        del model
        release_gpu_memory()


def describe_devices() -> str:
    """One-line accelerator summary for run logs and experiment manifests."""
    try:
        import torch
    except Exception:
        return "torch: not installed"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__} | device: cpu"
    props = torch.cuda.get_device_properties(0)
    return (f"torch {torch.__version__} | cuda {torch.version.cuda} | "
            f"{torch.cuda.device_count()}x {props.name} "
            f"({props.total_memory / 1e9:.1f} GB) | "
            f"ort: {','.join(available_onnx_providers()) or 'n/a'}")
