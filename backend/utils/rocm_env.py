"""Small pre-torch platform checks for ROCm environment setup."""


def should_probe_rocminfo(platform: str) -> bool:
    """Return whether ROCm discovery is relevant on this platform."""
    return platform != "darwin"


__all__ = ["should_probe_rocminfo"]
