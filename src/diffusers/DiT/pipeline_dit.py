try:
    from .pipeline import DiTPipeline
except ImportError:
    from pipeline import DiTPipeline

__all__ = ["DiTPipeline"]
