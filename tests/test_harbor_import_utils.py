import importlib.metadata

from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata


def test_ensure_harbor_distribution_metadata_adds_harbor_only_fallback(monkeypatch):
    def fake_version(name):
        if name == "harbor":
            raise importlib.metadata.PackageNotFoundError(name)
        if name == "other":
            return "1.2.3"
        raise AssertionError(f"unexpected distribution lookup: {name}")

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    ensure_harbor_distribution_metadata()

    assert importlib.metadata.version("harbor") == "0+slime-src"
    assert importlib.metadata.version("other") == "1.2.3"
