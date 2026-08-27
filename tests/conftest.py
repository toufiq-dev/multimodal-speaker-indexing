"""Test bootstrap.

The unit tests exercise pure logic (regexes, temporal assignment rules,
fusion construction, metrics, manifest schemas) and must run on any machine
-- including CI runners -- WITHOUT the multi-GB ML stack. If the real
libraries are importable we use them; otherwise we install lightweight
stand-ins so the engine modules can be imported. Model-loading functions are
never invoked in unit tests.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _try_import(name: str):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if not _try_import("torch"):
    torch = _module("torch")
    _cuda = _module("torch.cuda")
    _cuda.is_available = lambda: False
    torch.cuda = _cuda
    _backends = _module("torch.backends")
    _mps = _module("torch.backends.mps")
    _mps.is_available = lambda: False
    _backends.mps = _mps
    torch.backends = _backends
    torch.empty_cache = lambda: None
    torch.device = lambda *a, **k: None
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None,
                                                  __exit__=lambda s, *a: None)

if not _try_import("faster_whisper"):
    fw = _module("faster_whisper")

    class _WhisperModel:  # pragma: no cover - never constructed in tests
        def __init__(self, *a, **k):
            raise RuntimeError("Heavy model stack not installed; "
                               "unit tests do not load models.")

    fw.WhisperModel = _WhisperModel

if not _try_import("transformers"):
    tr = _module("transformers")
    for cls in ("WhisperProcessor", "WhisperForConditionalGeneration",
                "AutoTokenizer", "AutoModelForTokenClassification"):
        setattr(tr, cls, type(cls, (), {"from_pretrained": None}))
    tr.pipeline = None

if not _try_import("peft"):
    peft = _module("peft")
    peft.PeftModel = type("PeftModel", (), {"from_pretrained": None})

if not _try_import("torchaudio"):
    ta = _module("torchaudio")
    ta.load = None
    ta.transforms = types.SimpleNamespace(Resample=None)

if not _try_import("cv2"):
    cv2 = _module("cv2")
    cv2.imread = None
    cv2.resize = None
    cv2.absdiff = None

if not _try_import("insightface"):
    ins = _module("insightface")
    app = _module("insightface.app")
    fa = _module("insightface.app.FaceAnalysis")
    fa.FaceAnalysis = type("FaceAnalysis", (), {})
    app.FaceAnalysis = fa.FaceAnalysis
    ins.app = app

if not _try_import("sklearn"):
    sk = _module("sklearn")
    cl = _module("sklearn.cluster")
    cl.DBSCAN = type("DBSCAN", (), {})
    sk.cluster = cl

if not _try_import("pyannote.audio"):
    pa = _module("pyannote.audio")
    pa.Pipeline = type("Pipeline", (), {"from_pretrained": None})
