"""Core contracts: swappable-step protocols and pipeline configuration."""

from ottoparse.core.config import PipelineConfig
from ottoparse.core.strategy import ImageStep, PageRouter, RouteDecision

__all__ = ["PipelineConfig", "ImageStep", "PageRouter", "RouteDecision"]
