"""MiniCPM-o-4.5 test compatibility configuration."""

from __future__ import annotations

import pkgutil


# Compatibility shim for audioread/librosa on Python 3.12: audioread 3.x still
# registers pkgutil.ImpImporter which was removed in 3.12. Importing librosa
# (pulled in by the MiniCPM-O processor and several tests) fails without it.
# The finder is never exercised on Python 3; the shim only lets the import
# succeed.
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = type("ImpImporter", (), {})
