from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cues.level import (
    active_bins,
    azimuth_from_ild_db,
    ild_db_for_azimuth,
)


AZIMUTH_TOLERANCE_DEG = 5.0
MIN_PEAK_SEPARATION_DB = ild_db_for_azimuth(AZIMUTH_TOLERANCE_DEG)


ILD_GRID_STEP_DB = 0.25


ILD_GRID_LIMIT_DB = 60.0


SMOOTHING_BANDWIDTH_DB = 1.0


MIN_ENERGY_FRACTION = 0.05


@dataclass(frozen=True)
class SourcePeak:
    ild_db: float
    azimuth_deg: float
    energy_fraction: float
    bins: int


@dataclass(frozen=True)
class MixtureLevelEstimate:
    sources: tuple[SourcePeak, ...]
    active_bins: int
    grid_ild_db: np.ndarray
    density: np.ndarray

    @property
    def azimuths_deg(self) -> tuple[float, ...]:
        return tuple(source.azimuth_deg for source in self.sources)

    @property
    def source_count(self) -> int:
        return len(self.sources)


def ild_density(
    ild_db: np.ndarray,
    energy: np.ndarray,
    grid_step_db: float = ILD_GRID_STEP_DB,
    grid_limit_db: float = ILD_GRID_LIMIT_DB,
    bandwidth_db: float = SMOOTHING_BANDWIDTH_DB,
) -> tuple[np.ndarray, np.ndarray]:
    if ild_db.size == 0:
        raise ValueError("no active bins to build a density from")
    if grid_step_db <= 0.0 or bandwidth_db <= 0.0:
        raise ValueError("grid_step_db and bandwidth_db must be positive")

    grid = np.arange(-grid_limit_db, grid_limit_db + grid_step_db, grid_step_db)
    clamped = np.clip(ild_db, grid[0], grid[-1])
    counts, _ = np.histogram(
        clamped,
        bins=np.append(grid - grid_step_db / 2.0, grid[-1] + grid_step_db / 2.0),
        weights=energy,
    )


    radius = max(1, int(round(3.0 * bandwidth_db / grid_step_db)))
    offsets = np.arange(-radius, radius + 1) * grid_step_db
    kernel = np.exp(-0.5 * (offsets / bandwidth_db) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(counts, kernel, mode="same")
    return grid, density


def find_peaks(
    grid: np.ndarray,
    density: np.ndarray,
    min_separation_db: float = MIN_PEAK_SEPARATION_DB,
) -> list[int]:
    interior = np.arange(1, density.size - 1)
    local = interior[
        (density[1:-1] >= density[:-2]) & (density[1:-1] > density[2:])
    ]
    if density.size and density[0] > density[1]:
        local = np.append(local, 0)
    if density.size > 1 and density[-1] > density[-2]:
        local = np.append(local, density.size - 1)

    ordered = sorted(local, key=lambda index: float(density[index]), reverse=True)
    kept: list[int] = []
    for index in ordered:
        if all(
            abs(grid[index] - grid[other]) >= min_separation_db for other in kept
        ):
            kept.append(int(index))
    return kept


def estimate_mixture_level_cues(
    stereo: np.ndarray,
    sample_rate: int,
    max_sources: int | None = None,
    min_energy_fraction: float = MIN_ENERGY_FRACTION,
) -> MixtureLevelEstimate:
    bins = active_bins(stereo, sample_rate, per_frame=True)
    ild_db = bins.ild_db
    energy = bins.energy
    total_energy = float(energy.sum())
    if total_energy <= 0.0:
        raise ValueError("active bins carry no energy")

    grid, density = ild_density(ild_db, energy)
    peaks = find_peaks(grid, density)

    half_window = MIN_PEAK_SEPARATION_DB / 2.0
    sources: list[SourcePeak] = []
    for index in peaks:
        centre = float(grid[index])
        member = np.abs(ild_db - centre) <= half_window
        member_energy = float(energy[member].sum())
        fraction = member_energy / total_energy
        if fraction < min_energy_fraction or member_energy <= 0.0:
            continue


        refined = float(np.average(ild_db[member], weights=energy[member]))
        sources.append(
            SourcePeak(
                ild_db=refined,
                azimuth_deg=azimuth_from_ild_db(refined),
                energy_fraction=fraction,
                bins=int(np.count_nonzero(member)),
            )
        )

    sources.sort(key=lambda source: source.energy_fraction, reverse=True)
    if max_sources is not None:
        sources = sources[:max_sources]

    return MixtureLevelEstimate(
        sources=tuple(sources),
        active_bins=bins.count,
        grid_ild_db=grid,
        density=density,
    )
