"""Compatibility shim for the packaged Silverman test API."""

from silverman.silverman_test import SilvermanTestResult, silverman_test

__all__ = ["SilvermanTestResult", "silverman_test"]
