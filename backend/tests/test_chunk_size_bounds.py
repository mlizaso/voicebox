"""Invalid chunk bounds must fail promptly instead of looping indefinitely."""

import subprocess
import sys


def test_nonpositive_chunk_size_fails_promptly():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from backend.utils.chunked_tts import split_text_into_chunks\n"
            "for size in (0, -1, -100):\n"
            "    try:\n"
            "        split_text_into_chunks('Hello world', size)\n"
            "    except ValueError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError(f'Accepted invalid size {size}')\n",
        ],
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
