"""Source compatibility identity for preparation and package rebuild metadata."""

from pathlib import Path


def source_identity():
    """Digest of the Python and native sources every rank must run identically."""
    from importlib import util
    import flare_heliopause

    # Locate FlashInfer without importing it: CUDA device selection must precede PyTorch's import.
    flashinfer = util.find_spec("flashinfer")
    if flashinfer is None:
        raise RuntimeError("the preparation environment requires FlashInfer")
    root = Path(__file__).resolve().parents[2]
    return flare_heliopause.export_source_identity([
        root / "ignition", root / "helios" / "src", root / "heliopause" / "src",
        Path(flashinfer.origin).resolve().parent])
