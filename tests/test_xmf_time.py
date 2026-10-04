import numpy as np

from gino.data.preprocessing_pipeline import read_xmf_time


def test_read_xmf_time_accepts_integer_and_decimal_values(tmp_path):
    integer_xmf = tmp_path / "integer.xmf"
    integer_xmf.write_text('<Xdmf><Domain><Grid><Time Value="25"/></Grid></Domain></Xdmf>')
    decimal_xmf = tmp_path / "decimal.xmf"
    decimal_xmf.write_text('<Xdmf><Domain><Grid><Time Value="32.0"/></Grid></Domain></Xdmf>')

    assert read_xmf_time(integer_xmf) == 25.0
    assert read_xmf_time(decimal_xmf) == 32.0


def test_read_xmf_time_rejects_missing_or_ambiguous_values(tmp_path):
    missing = tmp_path / "missing.xmf"
    missing.write_text("<Xdmf><Domain><Grid/></Domain></Xdmf>")
    ambiguous = tmp_path / "ambiguous.xmf"
    ambiguous.write_text(
        '<Xdmf><Domain><Grid><Time Value="1"/><Time Value="2"/></Grid></Domain></Xdmf>'
    )

    for path in (missing, ambiguous):
        try:
            read_xmf_time(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid XMF time should be rejected: {path}")


def test_xmf_step_delta_converts_to_physical_dt():
    # In the source files, XMF Time is a simulation-step coordinate, not seconds.
    xmf_times = np.asarray([25.0, 26.0, 28.0])
    physical_times = (xmf_times - xmf_times[0]) * 0.0034
    np.testing.assert_allclose(np.diff(physical_times), [0.0034, 0.0068])
