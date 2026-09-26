from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RolloutContext(Mapping):
    """Immutable metadata for one input-pair transition in a simulation case."""
    case: str
    pair_id: Optional[int]
    frame_t: str
    frame_tp1: str
    phase_t: float
    phase_tp1: float
    phase_delta: float
    dt: float
    vtk_path: str
    vtk_path_tp1: str
    aoa_deg: float = 0.0
    freestream: tuple = (0.0, 0.0, 0.0)
    correspondence_source: str = "canonical_row_index"

    @classmethod
    def from_mapping(cls, value, pair_id=None):
        get = value.get
        return cls(
            case=str(get("case", "unknown")),
            pair_id=int(get("pair_id", pair_id)) if get("pair_id", pair_id) is not None else None,
            frame_t=str(get("frame_t", "")), frame_tp1=str(get("frame_tp1", "")),
            phase_t=float(get("phase_t", 0.0)), phase_tp1=float(get("phase_tp1", get("phase_t", 0.0))),
            phase_delta=float(get("phase_delta", 0.0)), dt=float(get("dt", 0.0034)),
            vtk_path=str(get("vtk_path", "")), vtk_path_tp1=str(get("vtk_path_tp1", "")),
            aoa_deg=float(get("aoa_deg", 0.0)),
            freestream=tuple(float(x) for x in get("freestream", (0.0, 0.0, 0.0))),
            correspondence_source=str(get("correspondence_source", "canonical_row_index")),
        )

    def advance_terminal(self):
        """Advance only when there is no known following pair in the sampled sequence."""
        return RolloutContext(
            case=self.case, pair_id=None, frame_t=self.frame_tp1, frame_tp1=self.frame_tp1,
            phase_t=self.phase_tp1, phase_tp1=self.phase_tp1, phase_delta=0.0, dt=self.dt,
            vtk_path=self.vtk_path_tp1, vtk_path_tp1=self.vtk_path_tp1,
            aoa_deg=self.aoa_deg, freestream=self.freestream,
            correspondence_source=self.correspondence_source,
        )

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(("case", "pair_id", "frame_t", "frame_tp1", "phase_t", "phase_tp1",
                     "phase_delta", "dt", "vtk_path", "vtk_path_tp1", "aoa_deg",
                     "freestream", "correspondence_source"))

    def __len__(self):
        return 13
