"""Extra experiment utilities for grid search and related searches."""

__all__ = [
    "ObservationConfig",
    "collect_dashboard_data",
    "default_search_space",
    "load_status",
    "run_grid_search",
    "run_line_search",
]


def __getattr__(name: str):
    if name == "ObservationConfig":
        from .grid_search import ObservationConfig

        return ObservationConfig
    if name == "default_search_space":
        from .grid_search import default_search_space

        return default_search_space
    if name == "run_grid_search":
        from .grid_search import run_grid_search

        return run_grid_search
    if name == "collect_dashboard_data":
        from .grid_search_dashboard import collect_dashboard_data

        return collect_dashboard_data
    if name == "load_status":
        from .grid_search_status import load_status

        return load_status
    if name == "run_line_search":
        from .line_search import run_line_search

        return run_line_search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
