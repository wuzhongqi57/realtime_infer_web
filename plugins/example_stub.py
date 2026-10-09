"""Example camera-control plugin (not loaded automatically).

Copy to local.py and implement available() / ffc_update() for your camera.
This stub never enables the toolbar button.
"""

NEEDS_DEVICE_RELEASE = True


def available() -> tuple[bool, str]:
    return False, "example stub is not a real camera-control plugin"


def ffc_update() -> None:
    raise RuntimeError("example stub cannot run camera control")
