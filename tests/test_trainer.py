import pytest

from wakeword_forge.trainer import SUPPORTED_BACKENDS, validate_backend


def test_public_release_supports_only_dscnn_backend():
    assert SUPPORTED_BACKENDS == {"dscnn"}
    assert validate_backend("dscnn") == "dscnn"

    with pytest.raises(ValueError, match="Valid options: dscnn"):
        validate_backend("legacy_backend")

    with pytest.raises(ValueError, match="Valid options: dscnn"):
        validate_backend("research_backend")
