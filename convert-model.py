import glob
import os

import valohai
from transformers import AutoModelForCausalLM, AutoTokenizer


def already_safe(model_dir: str) -> bool:
    """Return True if the directory already has safetensors files."""
    return (
        bool(glob.glob(os.path.join(model_dir, "*.safetensors"))) or
        os.path.exists(os.path.join(model_dir, "model.safetensors.index.json"))
    )

src = "nferruz/ProtGPT2"
dst = valohai.outputs("my-output").path("ProtGPT2-safetensors")

os.makedirs(dst, exist_ok=True)

if already_safe(dst):
    print(f"[SKIP] Safetensors already exist at {dst}")
else:
    print(f"[CONVERT] Converting {src} → {dst}")
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype="auto")

    tok.save_pretrained(dst)
    model.save_pretrained(dst, safe_serialization=True)
    print(f"[OK] Saved safetensors to {dst}")
