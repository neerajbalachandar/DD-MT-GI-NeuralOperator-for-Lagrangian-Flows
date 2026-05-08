"""Generic VTK loader for legacy and XML formats.

Supports automatic point/cell array discovery and scalar/vector/tensor classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ArrayInfo:
    """Metadata descriptor for one VTK array (association, shape, components, and semantic kind)."""
    name: str
    association: str  # "point" | "cell" | "field"
    dtype: str
    shape: Tuple[int, ...]
    components: int
    kind: str  # "scalar" | "vector" | "tensor" | "array"


@dataclass
class VTKData:
    """Normalized container for parsed VTK geometry, connectivity, and all discovered data arrays."""
    path: Path
    dataset_type: str
    points: np.ndarray
    cells: Dict[str, np.ndarray] = field(default_factory=dict)
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: Dict[str, np.ndarray] = field(default_factory=dict)
    field_data: Dict[str, np.ndarray] = field(default_factory=dict)
    arrays: Dict[str, ArrayInfo] = field(default_factory=dict)
    bounds: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    raw_mesh: Optional[Any] = None


def _classify_array(arr: np.ndarray) -> Tuple[int, str]:
    """Infer whether an array behaves like scalar/vector/tensor from its trailing component size."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return 1, "scalar"
    if arr.ndim >= 2:
        c = int(arr.shape[-1])
        if c == 1:
            return 1, "scalar"
        if c == 3:
            return 3, "vector"
        if c == 9:
            return 9, "tensor"
        return c, "array"
    return 1, "array"


def _register_arrays(dst: Dict[str, ArrayInfo], arrays: Dict[str, np.ndarray], association: str) -> None:
    """Register array metadata into the summary dictionary for debug and auditing."""
    for name, arr in arrays.items():
        a = np.asarray(arr)
        comps, kind = _classify_array(a)
        dst[name] = ArrayInfo(
            name=name,
            association=association,
            dtype=str(a.dtype),
            shape=tuple(int(s) for s in a.shape),
            components=comps,
            kind=kind,
        )


def _load_with_pyvista(path: Path) -> VTKData:
    """Load a VTK file through PyVista and extract points/cells/point-data/cell-data/field-data."""
    import pyvista as pv  # type: ignore

    mesh = pv.read(str(path))
    points = np.asarray(mesh.points) if hasattr(mesh, "points") else np.empty((0, 3), dtype=float)

    cells: Dict[str, np.ndarray] = {}
    for key in ("cells", "celltypes", "offset", "faces", "lines", "strips"):
        if hasattr(mesh, key):
            val = getattr(mesh, key)
            if val is not None:
                cells[key] = np.asarray(val)

    point_data = {name: np.asarray(mesh.point_data[name]) for name in mesh.point_data.keys()}
    cell_data = {name: np.asarray(mesh.cell_data[name]) for name in mesh.cell_data.keys()}
    field_data = {name: np.asarray(mesh.field_data[name]) for name in mesh.field_data.keys()}

    arrays: Dict[str, ArrayInfo] = {}
    _register_arrays(arrays, point_data, "point")
    _register_arrays(arrays, cell_data, "cell")
    _register_arrays(arrays, field_data, "field")

    b = tuple(float(v) for v in mesh.bounds)

    return VTKData(
        path=path,
        dataset_type=mesh.__class__.__name__,
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        arrays=arrays,
        bounds=(b[0], b[1], b[2], b[3], b[4], b[5]),
        raw_mesh=mesh,
    )


def _vtk_reader_for_suffix(suffix: str):
    """Select the appropriate VTK reader class based on file extension."""
    import vtk  # type: ignore

    suffix = suffix.lower()
    if suffix == ".vtk":
        return vtk.vtkGenericDataObjectReader()
    if suffix == ".vtu":
        return vtk.vtkXMLUnstructuredGridReader()
    if suffix == ".vtp":
        return vtk.vtkXMLPolyDataReader()
    if suffix == ".vtr":
        return vtk.vtkXMLRectilinearGridReader()
    if suffix == ".vts":
        return vtk.vtkXMLStructuredGridReader()
    if suffix == ".vti":
        return vtk.vtkXMLImageDataReader()
    raise ValueError(f"Unsupported VTK suffix: {suffix}")


def _vtk_array_dict(data_obj, association: str) -> Dict[str, np.ndarray]:
    """Convert VTK point/cell/field data arrays to numpy arrays keyed by array name."""
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

    if association == "point":
        data = data_obj.GetPointData()
    elif association == "cell":
        data = data_obj.GetCellData()
    else:
        data = data_obj.GetFieldData()

    out: Dict[str, np.ndarray] = {}
    if data is None:
        return out

    for i in range(data.GetNumberOfArrays()):
        arr = data.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName() or f"unnamed_{association}_{i}"
        out[name] = vtk_to_numpy(arr)
    return out


def _load_with_vtk(path: Path) -> VTKData:
    """Load a VTK file through raw VTK API when PyVista is unavailable or fails."""
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

    reader = _vtk_reader_for_suffix(path.suffix)
    reader.SetFileName(str(path))
    reader.Update()

    data_obj = reader.GetOutputDataObject(0)
    if data_obj is None:
        raise RuntimeError(f"Could not parse VTK file: {path}")

    points = np.empty((0, 3), dtype=float)
    if hasattr(data_obj, "GetPoints") and data_obj.GetPoints() is not None:
        points_vtk = data_obj.GetPoints().GetData()
        points = vtk_to_numpy(points_vtk).reshape(-1, 3)

    cells: Dict[str, np.ndarray] = {}
    if hasattr(data_obj, "GetCells") and data_obj.GetCells() is not None:
        cell_array = data_obj.GetCells().GetData()
        cells["cells"] = vtk_to_numpy(cell_array)
    if hasattr(data_obj, "GetCellTypesArray") and data_obj.GetCellTypesArray() is not None:
        cells["celltypes"] = vtk_to_numpy(data_obj.GetCellTypesArray())

    point_data = _vtk_array_dict(data_obj, "point")
    cell_data = _vtk_array_dict(data_obj, "cell")
    field_data = _vtk_array_dict(data_obj, "field")

    arrays: Dict[str, ArrayInfo] = {}
    _register_arrays(arrays, point_data, "point")
    _register_arrays(arrays, cell_data, "cell")
    _register_arrays(arrays, field_data, "field")

    if hasattr(data_obj, "GetBounds") and data_obj.GetBounds() is not None:
        b = data_obj.GetBounds()
        bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]))
    else:
        bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return VTKData(
        path=path,
        dataset_type=data_obj.GetClassName(),
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        arrays=arrays,
        bounds=bounds,
        raw_mesh=data_obj,
    )


def load_vtk(path: str | Path, prefer: str = "pyvista") -> VTKData:
    """Load VTK data with automatic backend fallback.

    Parameters
    ----------
    path
        Input `.vtk`, `.vtu`, `.vtp`, `.vtr`, `.vts`, or `.vti` file.
    prefer
        Either "pyvista" or "vtk". Whichever is selected is attempted first.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    backends = ["pyvista", "vtk"] if prefer == "pyvista" else ["vtk", "pyvista"]
    errs = []

    for backend in backends:
        try:
            if backend == "pyvista":
                return _load_with_pyvista(p)
            return _load_with_vtk(p)
        except Exception as exc:  # pragma: no cover
            errs.append((backend, str(exc)))

    formatted = "\n".join(f"- {b}: {e}" for b, e in errs)
    raise RuntimeError(
        f"Failed to load VTK file with all available backends: {p}\n{formatted}"
    )


def summarize_vtk_arrays(vtk_data: VTKData) -> Dict[str, Any]:
    """Compact array summary for logging/debug."""
    return {
        "path": str(vtk_data.path),
        "dataset_type": vtk_data.dataset_type,
        "n_points": int(vtk_data.points.shape[0]),
        "bounds": vtk_data.bounds,
        "point_arrays": sorted(vtk_data.point_data.keys()),
        "cell_arrays": sorted(vtk_data.cell_data.keys()),
        "field_arrays": sorted(vtk_data.field_data.keys()),
        "array_info": {
            name: {
                "association": info.association,
                "shape": info.shape,
                "components": info.components,
                "kind": info.kind,
            }
            for name, info in vtk_data.arrays.items()
        },
    }
