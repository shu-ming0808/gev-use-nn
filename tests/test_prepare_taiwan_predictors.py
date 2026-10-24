import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from taiwan_simulation import match_annual_events as io
from taiwan_simulation import prepare_predictors as p
from taiwan_simulation.prepare_static_predictors import class_fractions
from test_match_annual_events import write_registry, write_daily


def arco_file(path, date, grids):
    write_daily(path, lat=grids.latitude.unique(), lon=grids.longitude.unique(), dates=(date,))
    with nc.Dataset(path, 'a') as ds:
        ds.processing_contract = p.ARCO_CONTRACT
        ds.complete = 1
        ds.source = p.ARCO_STORE
        ds.date_utc = date
        ds.processing_metrics = json.dumps({'hourly_chunks': 120})
        for name, (stat, _) in io.FIELDS.items():
            ds[name].hourly_samples = 24
            ds[name].source_hour_window_utc = '01:00 through next-day 00:00' if stat == 'sum' else '00:00 through 23:00'
        ds['tcc'][:] = .5
        ds['tcc'].units = '(0 - 1)'
    path.with_suffix('.audit.json').write_text(json.dumps({
        'check': 'all_values', 'date_utc': date, 'source': p.ARCO_STORE,
        'file_bytes': path.stat().st_size, 'hourly_chunks': 120}), encoding='utf-8')


def cds_files(root, year, month, grids):
    dates = pd.date_range(f'{year}-{month:02d}-01', periods=io.calendar.monthrange(year, month)[1]).strftime('%Y-%m-%d').tolist()
    for name in ('u10', 'tp'):
        path = p.cds_path(root, year, month, name)
        write_daily(path, lat=grids.latitude.unique(), lon=grids.longitude.unique(), dates=dates)
        path.with_suffix('.request.json').write_text(json.dumps({'request': {
            'daily_statistic': 'daily_'+io.FIELDS[name][0], 'time_zone': 'utc+00:00', 'frequency': '1_hourly'}}), encoding='utf-8')


def test_arco_units_and_upstream_audit(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    path = p.arco_path(tmp_path, pd.Timestamp('2000-01-01'))
    arco_file(path, '2000-01-01', grids)
    a,q,c,errors = p.read_fields(path, grids, ['2000-01-01'], p.NAMES, 'arco')
    assert not errors and (q == 0).all() and (c == 1).all()
    assert a[0,0].tolist() == pytest.approx([3,4,.5,2,2])
    with nc.Dataset(path, 'a') as ds:
        ds['tp'].source_hour_window_utc = '00:00 through 23:00'
    with pytest.raises(ValueError):
        p.read_fields(path, grids, ['2000-01-01'], p.NAMES, 'arco')


def test_arco_bare_hourly_count_is_not_proof(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    path = p.arco_path(tmp_path, pd.Timestamp('2000-01-01'))
    arco_file(path, '2000-01-01', grids)
    audit = json.loads(path.with_suffix('.audit.json').read_text())
    audit['check'] = 'structure_and_samples'
    path.with_suffix('.audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='audit'):
        p.read_fields(path, grids, ['2000-01-01'], p.NAMES, 'arco')


def test_mixed_month_routes_dates_and_resume_invalidates_on_new_source(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    cds_files(tmp_path, 2000, 1, grids)
    path = p.arco_path(tmp_path, pd.Timestamp('2000-01-02'))
    arco_file(path, '2000-01-02', grids)
    report = p.extract_month(tmp_path, grids, 2000, 1, tmp_path/'cache')
    assert report['n_invalid_values'] == 0
    assert report['n_arco_variable_days'] == 5
    assert report['n_cds_variable_days'] == 150
    assert p.extract_month(tmp_path, grids, 2000, 1, tmp_path/'cache')['reused']
    with np.load(tmp_path/'cache/200001.npz') as data:
        assert (data['coverage'][0] == 2).all()
        assert (data['coverage'][1] == 1).all()
    arco_file(p.arco_path(tmp_path, pd.Timestamp('2000-01-03')), '2000-01-03', grids)
    report = p.extract_month(tmp_path, grids, 2000, 1, tmp_path/'cache')
    assert not report['reused'] and report['n_arco_variable_days'] == 10


def test_invalid_arco_not_hidden_by_valid_cds(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    cds_files(tmp_path, 2000, 1, grids)
    path = p.arco_path(tmp_path, pd.Timestamp('2000-01-02'))
    path.parent.mkdir(parents=True)
    path.write_bytes(b'not netcdf')
    report = p.extract_month(tmp_path, grids, 2000, 1, tmp_path/'cache')
    assert report['n_invalid_values'] == len(grids)*5
    with np.load(tmp_path/'cache/200001.npz') as data:
        assert (data['quality'][1] == 2).all()


def test_missing_month_preserved_in_cache(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    report = p.extract_month(tmp_path, grids, 2000, 2, tmp_path/'cache')
    assert report['n_days'] == 29
    assert report['n_invalid_values'] == 29*2*5


def test_slope_per_decade_and_missing_no_imputation():
    years = np.arange(1976,2026)
    t = (years-2000.5)/10
    assert p.slope(8+2*t, years) == pytest.approx(2)
    values = 8+2*t
    values[0] = np.nan
    assert np.isnan(p.slope(values, years))


def test_area_fractions_and_missing_cells_not_zero_filled():
    values = np.array([[10,50],[190,210]])
    result = class_fractions(values, np.array([[0,1],[1,2]]), np.array([[120,121],[121,122]]), [0,10,50,190,210])
    assert result['land_cover_valid_fraction'] == 1
    assert sum(result[f'{n}_ratio'] for n in ['agriculture','forest','urban','water','other']) == pytest.approx(1)
    values[0,0] = 0
    result = class_fractions(values, np.array([[0,1],[1,2]]), np.array([[120,121],[121,122]]), [0,10,50,190,210])
    assert result['land_cover_valid_fraction'] < 1
    assert result['agriculture_ratio'] == 0


def test_complete_year_summary_and_event_join(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    cache = tmp_path/'cache'; cache.mkdir()
    out = tmp_path/'out'; out.mkdir()
    years = [2000,2001]
    for y in years:
        for m in range(1,13):
            dates = pd.date_range(f'{y}-{m:02d}-01', periods=io.calendar.monthrange(y,m)[1]).strftime('%Y-%m-%d')
            a = np.ones((len(dates),2,5))
            a[:,:,0] = 3; a[:,:,1] = 4; a[:,:,2] = .5
            a[:,:,3] = 2 if y == 2000 else 3
            a[:,:,4] = 10
            np.savez_compressed(cache/f'{y}{m:02d}.npz', dates=np.asarray(dates,dtype=str),
                grid_id=grids.grid_id.to_numpy(dtype=str),variables=np.asarray(p.NAMES),values=a,
                quality=np.zeros_like(a,dtype='u1'),coverage=np.ones_like(a,dtype='u1'),
                source_kind=np.ones((len(dates),5),dtype='u1'))
    events = pd.DataFrame([dict(grid_id=g,year=y,event_date_utc=f'{y}-07-01',event_status='ok',detail='')
                          for g in grids.grid_id for y in years])
    frames,_ = p.assemble(cache,grids,events,years,out,tmp_path)
    annual = frames['annual_weather.csv']
    assert annual.tp_annual_total.tolist() == [732,732,1095,1095]
    assert annual.wind_vector_magnitude_annual_mean.eq(5).all()
    summary = frames['annual_weather_summary.csv']
    assert summary.tp_annual_total_trend_per_decade.tolist() == pytest.approx([3630,3630])
    assert frames['event_weather_long.csv'].shape[0] == 20
    assert frames['event_grid_predictors.csv'].ready_for_spatial_cv.all()


def cached_year_with_issues(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    cache = tmp_path/'cache'; cache.mkdir()
    for month in range(1, 13):
        dates = pd.date_range(f'2000-{month:02d}-01', periods=io.calendar.monthrange(2000, month)[1]).strftime('%Y-%m-%d')
        values = np.ones((len(dates), 2, 5)); values[:, :, 2] = .5
        quality = np.zeros_like(values, dtype='u1')
        kinds = np.ones((len(dates), 5), dtype='u1')
        if month == 1:
            values[0:4, 0, 3] = [-.0001, -.00010001, np.nan, -.00004]
            quality[0:4, 0, 3] = [5, 5, 4, 5]
            kinds[1, 3] = 2
        np.savez_compressed(cache/f'2000{month:02d}.npz', dates=np.asarray(dates, dtype=str),
            grid_id=grids.grid_id.to_numpy(dtype=str), variables=np.asarray(p.NAMES), values=values,
            quality=quality, coverage=np.ones_like(quality), source_kind=kinds)
    events = pd.DataFrame([dict(grid_id=g, year=2000, event_date_utc='2000-01-01', event_status='ok', detail='')
                           for g in grids.grid_id])
    out = tmp_path/'out'; out.mkdir()
    return grids, cache, events, out


def test_numpy_date_issue_report_keeps_invalid_raw_values(tmp_path):
    grids, cache, events, out = cached_year_with_issues(tmp_path)
    frames, _ = p.assemble(cache, grids, events, [2000], out, tmp_path)
    issues = frames['daily_missing_report.csv']
    assert len(issues) == 4
    assert issues.iloc[0].raw_converted_value == -.0001
    assert 'arco_daily_predictors' in issues.iloc[0].source
    assert 'daily_predictors' in issues.iloc[1].source
    assert np.isnan(frames['annual_weather.csv'].iloc[0].tp_annual_total)
    assert frames['annual_weather.csv'].iloc[1].tp_annual_total == 366


def test_explicit_small_negative_cleaning_preserves_raw_and_real_missingness(tmp_path):
    grids, cache, events, out = cached_year_with_issues(tmp_path)
    before = {f.name: io.sha256(f) for f in cache.glob('*.npz')}
    frames, _ = p.assemble(cache, grids, events, [2000], out, tmp_path,
                          precipitation_zero_tolerance_mm=.0001)
    corrections = frames['precipitation_corrections.csv']
    assert corrections.raw_value_mm.tolist() == [-.0001, -.00004]
    assert corrections.cleaned_value_mm.eq(0).all()
    assert len(frames['daily_missing_report.csv']) == 2  # larger negative + NaN remain invalid
    annual = frames['annual_weather.csv']
    assert np.isnan(annual.iloc[0].tp_annual_total)
    assert annual.iloc[0].tp_zero_correction_total_mm == pytest.approx(.00014)
    paired = frames['event_weather.csv']
    assert paired.iloc[0].tp_value == 0 and paired.iloc[0].tp_zero_clipped
    assert paired.iloc[0].tp_raw_value == -.0001
    daily = pd.read_csv(out/'daily_taiwan/taiwan_daily_2000.csv.gz')
    assert daily.iloc[0].tp == 0 and daily.iloc[0].tp_raw == -.0001
    assert daily.iloc[0].tp_quality == 0 and daily.iloc[0].tp_quality_raw == 5
    assert np.isnan(daily.iloc[2].tp) and daily.iloc[2].tp_quality == 4
    assert before == {f.name: io.sha256(f) for f in cache.glob('*.npz')}


def test_cached_month_verification_is_read_only_and_rejects_changed_source(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    cds_files(tmp_path, 2000, 1, grids)
    cache = tmp_path/'cache'
    p.extract_month(tmp_path, grids, 2000, 1, cache)
    before = {f.name: io.sha256(f) for f in cache.iterdir()}
    assert p.verified_cached_month(cache, grids, 2000, 1)['reused']
    source = p.cds_path(tmp_path, 2000, 1, 'tp')
    with nc.Dataset(source, 'a') as ds:
        ds.diagnostic_change = 'changed'
    with pytest.raises(ValueError, match='source changed'):
        p.verified_cached_month(cache, grids, 2000, 1)
    assert before == {f.name: io.sha256(f) for f in cache.iterdir()}


def test_cached_month_verification_rejects_corrupt_cache(tmp_path):
    grids = write_registry(tmp_path/'grids.csv')
    cache = tmp_path/'cache'
    p.extract_month(tmp_path, grids, 2000, 1, cache)
    (cache/'200001.npz').write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError, match='hash mismatch'):
        p.verified_cached_month(cache, grids, 2000, 1)


@pytest.mark.parametrize('tolerance', [-1, np.nan, np.inf])
def test_invalid_precipitation_tolerance_rejected(tmp_path, tolerance):
    with pytest.raises(ValueError, match='tolerance'):
        p.assemble(None, None, None, None, tmp_path, tmp_path,
                   precipitation_zero_tolerance_mm=tolerance)

