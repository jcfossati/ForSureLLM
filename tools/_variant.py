"""Helper : extrait `--variant <suffix>` de sys.argv et set `FORSURELLM_VARIANT`
*avant* l'import de forsurellm. À importer en tout premier dans chaque outil
qui peut basculer entre la variante par défaut et la variante prunée.
"""
from __future__ import annotations

import os
import sys


def consume_variant_arg() -> str:
    """Retire --variant du sys.argv si présent, set l'env var, retourne la valeur."""
    args = sys.argv
    i = 1
    while i < len(args):
        a = args[i]
        if a == "--variant" and i + 1 < len(args):
            value = args[i + 1]
            del args[i:i + 2]
            os.environ["FORSURELLM_VARIANT"] = value
            return value
        if a.startswith("--variant="):
            value = a.split("=", 1)[1]
            del args[i]
            os.environ["FORSURELLM_VARIANT"] = value
            return value
        i += 1
    return os.environ.get("FORSURELLM_VARIANT", "")


# Auto-applied at import time
_VARIANT = consume_variant_arg()
