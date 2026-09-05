"""Avatar processing preserves appearance across camera orientations and modes."""

import pytest
from PIL import Image

from backend.utils.images import process_avatar


@pytest.mark.parametrize(
    ("orientation", "size", "pixels"),
    [
        (1, (3, 2), [10, 20, 30, 40, 50, 60]),
        (2, (3, 2), [30, 20, 10, 60, 50, 40]),
        (3, (3, 2), [60, 50, 40, 30, 20, 10]),
        (4, (3, 2), [40, 50, 60, 10, 20, 30]),
        (5, (2, 3), [10, 40, 20, 50, 30, 60]),
        (6, (2, 3), [40, 10, 50, 20, 60, 30]),
        (7, (2, 3), [60, 30, 50, 20, 40, 10]),
        (8, (2, 3), [30, 60, 20, 50, 10, 40]),
    ],
)
def test_avatar_applies_all_exif_orientations(tmp_path, orientation, size, pixels):
    source = tmp_path / "camera.png"
    output = tmp_path / "avatar.png"
    image = Image.new("L", (3, 2))
    image.putdata([10, 20, 30, 40, 50, 60])
    exif = Image.Exif()
    exif[274] = orientation
    image.save(source, exif=exif)

    process_avatar(str(source), str(output))

    with Image.open(output) as avatar:
        assert avatar.size == size
        assert list(avatar.tobytes()) == pixels
        assert avatar.getexif().get(274, 1) == 1


def test_avatar_resizes_and_flattens_transparency_on_white(tmp_path):
    source = tmp_path / "transparent.png"
    output = tmp_path / "avatar.png"
    Image.new("RGBA", (100, 50), (255, 0, 0, 128)).save(source)

    process_avatar(str(source), str(output), max_size=20)

    with Image.open(output) as avatar:
        assert avatar.mode == "RGB"
        assert avatar.size == (20, 10)
        assert avatar.getpixel((10, 5)) == (255, 127, 127)
