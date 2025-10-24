"""Weather visualization utilities for the WeatherGenius persona."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Mapping, Sequence


@dataclass(frozen=True)
class WeatherSeries:
    """Container describing a single plotted series."""

    label: str
    unit: str
    values: Sequence[float]


DEFAULT_DATES: Sequence[str] = (
    "2022-01-01",
    "2022-01-02",
    "2022-01-03",
    "2022-01-04",
    "2022-01-05",
)

DEFAULT_TEMPERATURES: Sequence[float] = (
    85.96,
    87.28,
    92.98,
    96.69,
    98.29,
)

DEFAULT_RAINFALL: Sequence[float] = (
    0.0,
    3.98,
    1.94,
    0.0,
    0.31,
)


async def render_weather_visualization(
    dates: Sequence[str] | None = None,
    temperatures: Sequence[float] | None = None,
    rainfall: Sequence[float] | None = None,
    *,
    backend: str = "Agg",
) -> Mapping[str, object]:
    """Render a combined temperature/rainfall plot.

    Parameters
    ----------
    dates:
        Labels to use along the x-axis. Defaults to a fixed Shreveport, LA
        forecast range when not provided.
    temperatures:
        Temperature values matched to ``dates``. Defaults to a fixed set when
        omitted.
    rainfall:
        Rainfall amounts matched to ``dates``. Defaults to a fixed set when
        omitted.
    backend:
        Optional Matplotlib backend override. ``Agg`` is used by default so the
        function can execute within headless environments without requiring a
        graphical display server.

    Returns
    -------
    Mapping[str, object]
        A serialisable dictionary containing the rendered image encoded as
        base64 alongside the underlying data for downstream consumers.
    """

    try:
        import matplotlib

        if backend:
            matplotlib.use(backend, force=True)

        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "matplotlib is required to render weather visualisations."
        ) from exc
    except Exception as exc:  # pragma: no cover - backend guard
        raise RuntimeError(
            "matplotlib backend initialisation failed for weather rendering."
        ) from exc

    resolved_dates = list(dates or DEFAULT_DATES)
    resolved_temperatures = list(temperatures or DEFAULT_TEMPERATURES)
    resolved_rainfall = list(rainfall or DEFAULT_RAINFALL)

    if not (len(resolved_dates) == len(resolved_temperatures) == len(resolved_rainfall)):
        raise ValueError("dates, temperatures, and rainfall must contain the same number of items")

    fig, ax1 = plt.subplots()

    ax1.plot(resolved_dates, resolved_temperatures, color="tab:red", marker="o")
    ax1.set_ylabel("Temperature (°F)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.bar(resolved_dates, resolved_rainfall, color="tab:blue", alpha=0.5)
    ax2.set_ylabel("Rainfall (inches)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Weather Forecast for Shreveport, LA")
    plt.xlabel("Date")

    ax1.legend(["Temperature"], loc="upper left")
    ax2.legend(["Rainfall"], loc="upper right")

    plt.xticks(rotation=45)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "title": "Weather Forecast for Shreveport, LA",
        "image_base64": encoded_image,
        "series": [
            asdict(WeatherSeries(label="Temperature", unit="°F", values=resolved_temperatures)),
            asdict(WeatherSeries(label="Rainfall", unit="inches", values=resolved_rainfall)),
        ],
        "axis": {
            "x": list(resolved_dates),
        },
    }


__all__ = ["render_weather_visualization"]
