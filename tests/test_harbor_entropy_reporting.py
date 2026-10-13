from __future__ import annotations

import importlib

import pytest


train_loss = importlib.import_module("slime_plugins.harbor.train_loss")


class _Args:
    """Minimal args carrier; attribute may be absent to test env fallback."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_report_entropy_defaults_on_when_unset(monkeypatch):
    monkeypatch.delenv("HARBOR_REPORT_ENTROPY", raising=False)
    assert train_loss._should_report_entropy(_Args()) is True


@pytest.mark.parametrize("falsey", ["0", "false", "False", "no", "off", "OFF", ""])
def test_report_entropy_env_off_switch(monkeypatch, falsey):
    monkeypatch.setenv("HARBOR_REPORT_ENTROPY", falsey)
    assert train_loss._should_report_entropy(_Args()) is False


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "True"])
def test_report_entropy_env_on(monkeypatch, truthy):
    monkeypatch.setenv("HARBOR_REPORT_ENTROPY", truthy)
    assert train_loss._should_report_entropy(_Args()) is True


def test_report_entropy_args_attr_takes_precedence_over_env(monkeypatch):
    # Explicit args attribute wins over the env var.
    monkeypatch.setenv("HARBOR_REPORT_ENTROPY", "1")
    assert train_loss._should_report_entropy(_Args(harbor_report_entropy="0")) is False

    monkeypatch.setenv("HARBOR_REPORT_ENTROPY", "0")
    assert train_loss._should_report_entropy(_Args(harbor_report_entropy="1")) is True
