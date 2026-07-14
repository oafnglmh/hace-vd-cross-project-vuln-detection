"""Loss functions used by HACE-VD."""

from .contrastive import supervised_contrastive_loss
from .domain_loss import soft_cross_entropy
from .ensemble import weighted_vulnerability_loss

__all__ = [
    "supervised_contrastive_loss",
    "soft_cross_entropy",
    "weighted_vulnerability_loss",
]
