"""Pre-load torch before pdfplumber to avoid native library conflicts.

On macOS x86_64, pdfplumber → pdfminer → cryptography loads native libs
that conflict with PyTorch's OpenMP/MKL. Loading torch first avoids the
segfault. Import this module at the top of any entry-point script that
uses both PDF processing and torch-based models.

Usage:
    import benchmarks._preload  # noqa: F401  (must be first import)
"""

import os

# Allow duplicate OpenMP libraries (torch + system OpenMP on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Force torch to load its native libs before anything else
import torch  # noqa: F401, E402
