"""
SPX v6.2 [System Suite]
Module: spx_env
Role: Environment Validator.
Description: Dependency and JIT-status verification for the SPX 4-pillar engine.
"""

import sys
import logging
import importlib.metadata
from typing import List, Tuple

# Global Unified Logger
logger = logging.getLogger("zpng")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# [(Package Name, Requirement Version, Description)]
REQUIRED_PACKAGES: List[Tuple[str, str, str]] = [
    ("numpy", "1.22.0", "Numerical array processing"),
    ("numba", "0.57.0", "JIT compilation and acceleration"),
    ("zstandard", "0.19.0", "Entropy coding and header compression"),
    ("Pillow", "9.0.0", "Image I/O support")
]

def _compare_version(current: str, required: str) -> bool:
    c_parts = [int(p) for p in current.split('.')[:3]]
    r_parts = [int(p) for p in required.split('.')[:3]]
    return c_parts >= r_parts

def verify_environment() -> bool:
    """ 
    Validates that all external dependencies and versions are correctly configured.
    Raises SystemExit if a critical dependency is missing or outdated.
    """
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
        logger.error("Environment verification failed. Please run: pip install -U numpy numba zstandard Pillow")
        sys.exit(1)
        
    # Check Numba JIT status
    try:
        import numba
        if not numba.config.DISABLE_JIT:
            logger.debug("Numba JIT is enabled and functional.")
        else:
            logger.warning("Numba JIT is DISABLED. Performance will be severely impacted.")
    except Exception as e:
        logger.error(f"Numba configuration check failed: {e}")
        
    return True

if __name__ == "__main__":
    verify_environment()
    print("Environment OK")
