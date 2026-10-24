import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from global_pipeline import diagnose_eas_buffer_folds as eas
from taiwan_simulation.diagnose_buffer_folds import geometry_diagnostics


def test_geodesic_matrix_matches_wgs84_and_not_degrees():
    grids = pd.DataFrame({'longitude_180':[100.,120.,140.], 'latitude':[20.,30.,40.]})
    matrix = eas.geodesic_matrix(grids)
    expected = Geod(ellps='WGS84').inv(100,20,120,30)[2]/1000
    assert matrix[0,1] == pytest.approx(expected)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix),0)


def test_distance_override_controls_both_outer_and_inner_buffers():
    xy = np.array([(i,j) for i in range(4) for j in range(4)],dtype=float)
    grids = pd.DataFrame({'grid_id':[str(i) for i in range(16)],'x_km':xy[:,0],'y_km':xy[:,1]})
    # All distinct GRID are 100 km apart in this explicit test metric, not xy.
    distance = np.full((16,16),100.); np.fill_diagonal(distance,0)
    result = geometry_diagnostics(grids,outer_counts=[2],inner_counts=[2],buffers=[50.,101.],
        min_train=2,distance_matrix=distance)
    good = result['splits'].query('buffer_km == 50')
    assert good.min_train_pass.all()
    assert good.nearest_distance_km.eq(100).all()
    bad = result['splits'].query('buffer_km == 101')
    assert not bad.min_train_pass.any()
    assert bad.n_train.eq(0).all()


def test_invalid_distance_matrix_rejected():
    grids = pd.DataFrame({'grid_id':list('abcd'),'x_km':[0,1,2,3],'y_km':[0,1,2,3]})
    with pytest.raises(ValueError,match='Distance matrix'):
        geometry_diagnostics(grids,outer_counts=[2],inner_counts=[2],buffers=[0],distance_matrix=np.ones((3,3)))
