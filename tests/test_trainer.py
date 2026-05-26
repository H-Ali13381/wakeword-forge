import pytest

from forge.trainer import SUPPORTED_BACKENDS, validate_backend


def test_public_release_supports_only_wavlm_repcnn_backend():
    assert SUPPORTED_BACKENDS == {"wavlm-repcnn"}
    assert validate_backend("wavlm-repcnn") == "wavlm-repcnn"

    with pytest.raises(ValueError, match="Valid options: wavlm-repcnn"):
        validate_backend("dscnn")

    with pytest.raises(ValueError, match="Valid options: wavlm-repcnn"):
        validate_backend("research_backend")
