import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException
from app import validate_audio_file


def test_validate_audio_file_rejects_missing_or_invalid_filenames():
    with pytest.raises(HTTPException) as excinfo:
        validate_audio_file(None)

    assert excinfo.value.status_code == 400
