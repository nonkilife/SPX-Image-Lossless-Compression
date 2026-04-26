"""
SPX v1.0.0 [System Suite]
Module: spx_env
Role: Environment Validator.

Description: 
Dependency verification and configuration hub for the SPX 4-pillar engine. 
Ensures system stability by validating library versions and providing 
diagnostic hooks via environment variables.

Architecture & Engineering Rationale:
1. Native Extension: SPX requires the 'spx_rans' shared library (compiled via 
   Rust/Maturin) for all high-performance operations. This module checks 
   for core Python dependencies that bridge the native layer.
2. Diagnostic Hooks: Environment variables allow for deep auditing of the 
   codec's internal state (e.g., dumping shards) without modifying source code.

Supported Environment Variables:
------------------------------
- SPX_DUMP_SHARDS: [0|1] 
    Dumps raw shard residuals to binary files for diagnostic auditing.
- SPX_DISABLE_TEMPLATES: [0|1] 
    Forces Mode 0 (Custom PDF) for all shards, bypassing empirical templates. 
    Useful for testing the mathematical limit of the rANS engine.
- SPX_FORCE_BITPLANE: [0|1] 
    Forces the Bitplane rANS engine regardless of entropy gating.
- SPX_LOG_LEVEL: [DEBUG|INFO|WARNING|ERROR] 
    Sets the internal logger verbosity.
"""

__version__ = "1.0.0"

__all__ = ['verify_environment', 'REQUIRED_PACKAGES']

import sys
import logging
import importlib.metadata
from typing import List, Tuple

# Global Unified Logger
logger = logging.getLogger("spx")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# [(Package Name, Requirement Version, Description)]
REQUIRED_PACKAGES: List[Tuple[str, str, str]] = [
    ("numpy", "1.22.0", "Numerical array processing"),
    ("zstandard", "0.19.0", "Entropy coding and header compression"),
    ("Pillow", "9.0.0", "Image I/O support")
]

def _compare_version(current: str, required: str) -> bool:
    c_parts = [int(p) for p in current.split('.')[:3]]
    r_parts = [int(p) for p in required.split('.')[:3]]
    return c_parts >= r_parts

_verified: bool = False

def verify_environment() -> bool:
    """
    Validates that all external dependencies and versions are correctly configured.
    Raises SystemExit if a critical dependency is missing or outdated.
    Safe to call multiple times — runs only once per process.
    """
    global _verified
    if _verified:
        return True
    failed = False
    for pkg, req_v, desc in REQUIRED_PACKAGES:
        try:
            # Pillow is registered as 'Pillow' in metadata, not 'PIL'
            current_v = importlib.metadata.version(pkg)
            if not _compare_version(current_v, req_v):
                logger.error(f"Version Mismatch: {pkg} {current_v} is too old (Requires >={req_v})")
                failed = True
        except importlib.metadata.PackageNotFoundError:
            logger.error(f"Dependency Missing: {pkg} ({desc})")
            failed = True
            
    if failed:
        logger.error("Environment verification failed. Please run: pip install -U numpy zstandard Pillow")
        sys.exit(1)

    _verified = True
    return True

if __name__ == "__main__":
    verify_environment()
    print("Environment OK")
