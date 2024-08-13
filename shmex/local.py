"""
Edit this file to customize input and output locations.
"""

import os

LOCAL_PATHS = {
    "DATA_DIR": "~/data",
    "TRAINED_MODELS_DIR": "~/re/netam-experiments-1/train/trained_models",
    "TEST_OUTPUT_DIR": "~/re/netam-experiments-1/train/_ignore/test_output",
    "FIGURES_DIR": "~/writing/talks/figures/bcr-mut-sel/",
}


def localify(path):
    for key, value in LOCAL_PATHS.items():
        path = path.replace(key, value)
    path = path.replace("~", os.path.expanduser("~"))
    return path
