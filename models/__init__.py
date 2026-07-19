"""Neural network modules used by HACE-VD.

Note. ``CodePTMBackbone`` requires ``transformers`` which is heavy to install
in a smoke-test environment. We expose it lazily so that the rest of the
package can be imported independently.
"""

from .expert import DomainExpert
from .domain_classifier import DomainClassifier
from .projection_head import ProjectionHead
from .hacevd_model import HACEVDModel
from .fusion_hace import build_fusion



def __getattr__(name: str):
    if name == "CodePTMBackbone":
        from .backbone import CodePTMBackbone
        return CodePTMBackbone
    raise AttributeError(f"module 'models' has no attribute {name!r}")


__all__ = [
    "CodePTMBackbone",
    "DomainExpert",
    "DomainClassifier",
    "ProjectionHead",
    "HACEVDModel",
]
