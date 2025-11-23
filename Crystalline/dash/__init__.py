"""Dash-specific training utilities for Crystalline AugNODE models.

This module provides training functions optimized for integration with Dash apps,
including proper progress reporting and real-time updates.
"""

from .training import train_AugNODE_dash_realtime

__all__ = ["train_AugNODE_dash_realtime"]
