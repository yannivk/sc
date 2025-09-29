"""
Single-Cell RNA-seq Analysis Pipeline
Interactive pipeline for scRNA-seq preprocessing, clustering, and annotation
"""

__version__ = "0.1.0"
__author__ = "SC Pipeline Team"

from . import io
from . import qc
from . import preprocessing
from . import reduction
from . import clustering
from . import annotation
from . import visualization
from . import interactive
from . import reports

__all__ = [
    "io",
    "qc",
    "preprocessing",
    "reduction",
    "clustering",
    "annotation",
    "visualization",
    "interactive",
    "reports",
]