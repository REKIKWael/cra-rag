"""Smoke test: the package installs and imports. Real tests arrive with step 2."""

import cra_rag


def test_package_exposes_a_version():
    assert cra_rag.__version__
