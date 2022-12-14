"""Script to compute and apply a lapse rate correction, a bias correction and
reliability calibration to the EUPPBench benchmark datasets provided by the
EUMETNET project as a contribution to the Earth System Science Data (ESSD) journal."""

import argparse
import datetime
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cf_units
import iris
import numpy as np
import pandas as pd
import xarray as xr
from improver.cli import (
    apply_reliability_calibration,
    construct_reliability_tables,
    generate_realizations,
    manipulate_reliability_table,
    threshold,
)
from iris.cube import Cube, CubeList

warnings.filterwarnings("ignore", category=UserWarning)


def get_train_forecasts_and_observations(input_path: Path) -> List[CubeList]:
    """Function to load the EUPPBench training data using xarray
    then manipulate the arrays into iris cube format compatible with improver.

    Args:
        input_path: Path to the training dataset forecasts and observations.

    Returns:
        Returns a list, with first entry an Iris cubelist of training forecasts,
        and second entry an Iris cubelist of corresponding training observations.
    """
    fcs_tr = xr.open_dataarray(input_path / "ESSD_benchmark_training_data_forecasts.nc")
    obs_tr = xr.open_dataarray(
        input_path / "ESSD_benchmark_training_data_observations.nc"
    )

    fcs_tr.attrs["standard_name"] = "air_temperature"
    obs_tr.attrs["standard_name"] = "air_temperature"
    obs_tr.attrs["units"] = "K"
    # Reversing year coordinate, so that the most recent years have smaller values
    fcs_tr["year"] = len(fcs_tr["year"]) + 1 - fcs_tr["year"]
    obs_tr["year"] = len(obs_tr["year"]) + 1 - obs_tr["year"]
    fcs_tr = fcs_tr.sortby("station_id")
    obs_tr = obs_tr.sortby("station_id")
    fcs_tr = fcs_tr.assign_coords(wmo_id=("station_id", fcs_tr["station_id"].values))
    obs_tr = obs_tr.assign_coords(wmo_id=("station_id", fcs_tr["station_id"].values))
    fcs_tr = fcs_tr.assign_coords(station_id=np.arange(0, len(fcs_tr.station_id)))
    obs_tr = obs_tr.assign_coords(station_id=np.arange(0, len(obs_tr.station_id)))

    # combine 'year' and 'time' dimensions into one 'forecast_reference_time' dimension
    fcs_tr_new = fcs_tr.stack(frt=["year", "time"])
    frt = fcs_tr_new.time.to_index() - np.array(fcs_tr_new.year, dtype="timedelta64[Y]")
    frt = np.array(np.array(frt, dtype="datetime64[D]"), dtype="datetime64[ns]")
    fcs_tr_new = fcs_tr_new.assign_coords(frt=frt)

    obs_tr_new = obs_tr.stack(ort=["year", "time"])
    ort = obs_tr_new.time.to_index() - np.array(obs_tr_new.year, dtype="timedelta64[Y]")
    ort = np.array(np.array(ort, dtype="datetime64[D]"), dtype="datetime64[ns]")
    obs_tr_new = obs_tr_new.assign_coords(ort=ort)

    # rename remaining coordinates in line with Improver norms
    fcs_tr_new = fcs_tr_new.rename(
        {
            "step": "forecast_period",
            "frt": "forecast_reference_time",
            "number": "realization",
            "station_id": "spot_index",
        }
    )
    fcs_tr_new = fcs_tr_new.sortby("forecast_reference_time")
    obs_tr_new = obs_tr_new.rename(
        {
            "step": "forecast_period",
            "ort": "forecast_reference_time",
            "station_id": "spot_index",
        }
    )
    obs_tr_new = obs_tr_new.sortby("forecast_reference_time")

    # create cube lists of forecast training data, each cube in the list corresponds
    # with one forecast_period
    fcs_tr_cube_list = []
    for i in range(0, len(fcs_tr_new.forecast_period)):
        fcs_tr_temp = fcs_tr_new.isel(forecast_period=i)
        fcs_tr_temp = fcs_tr_temp.assign_coords(
            time=(
                fcs_tr_temp["forecast_reference_time"] + fcs_tr_temp["forecast_period"]
            )
        )
        fcs_tr_cube_temp = fcs_tr_temp.to_iris()
        fcs_tr_cube_temp.coord("forecast_reference_time").units = cf_units.Unit(
            fcs_tr_cube_temp.coord("forecast_reference_time").units.origin,
            calendar="gregorian",
        )
        fcs_tr_cube_temp.coord("time").units = cf_units.Unit(
            fcs_tr_cube_temp.coord("time").units.origin, calendar="gregorian"
        )
        # when converting from xarray to iris, coordinate names do not automatically
        # get assigned to "standard" or "long" names in the cube, the following
        # renaming fixes this issue
        fcs_tr_cube_temp.coord("station_latitude").rename("latitude")
        fcs_tr_cube_temp.coord("station_longitude").rename("longitude")
        fcs_tr_cube_temp.coord("spot_index").rename("spot_index")
        fcs_tr_cube_temp.coord("forecast_reference_time").rename(
            "forecast_reference_time"
        )
        fcs_tr_cube_temp.coord("forecast_period").rename("forecast_period")
        fcs_tr_cube_temp.coord("wmo_id").rename("wmo_id")
        fcs_tr_cube_list.append(fcs_tr_cube_temp.copy())
    fcs_tr_cube_list = iris.cube.CubeList(fcs_tr_cube_list)

    # create cube lists of observation training data, each cube in the list corresponds
    # with one forecast_period
    obs_tr_cube_list = []
    for i in range(0, len(obs_tr_new.forecast_period)):
        obs_tr_temp = obs_tr_new.isel(forecast_period=i)
        obs_tr_temp = obs_tr_temp.assign_coords(
            time=(
                obs_tr_temp["forecast_reference_time"] + obs_tr_temp["forecast_period"]
            )
        )
        obs_tr_cube_temp = obs_tr_temp.to_iris()
        obs_tr_cube_temp.coord("forecast_reference_time").units = cf_units.Unit(
            obs_tr_cube_temp.coord("forecast_reference_time").units.origin,
            calendar="gregorian",
        )
        obs_tr_cube_temp.coord("time").units = cf_units.Unit(
            obs_tr_cube_temp.coord("time").units.origin, calendar="gregorian"
        )
        # when converting from xarray to iris, coordinate names do not automatically
        # get assigned to "standard" or "long" names in the cube, the following
        # renaming fixes this issue
        obs_tr_cube_temp.coord("latitude").rename("latitude")
        obs_tr_cube_temp.coord("longitude").rename("longitude")
        obs_tr_cube_temp.coord("spot_index").rename("spot_index")
        obs_tr_cube_temp.coord("forecast_reference_time").rename(
            "forecast_reference_time"
        )
        obs_tr_cube_temp.coord("forecast_period").rename("forecast_period")
        obs_tr_cube_temp.coord("wmo_id").rename("wmo_id")
        obs_tr_cube_temp.attributes["truth"] = "truth"

        obs_tr_cube_list.append(obs_tr_cube_temp.copy())
    obs_tr_cube_list = iris.cube.CubeList(obs_tr_cube_list)

    return [fcs_tr_cube_list, obs_tr_cube_list]


def get_test_forecasts_and_observations(input_path: Path) -> List[CubeList]:
    """Function to load the EUPPBench test data using xarray
    then manipulate the arrays into iris cube format compatible with improver.

    Args:
        input_path: Path to the test dataset forecasts and observations.

    Returns:
        Returns a list, with first entry an Iris cubelist of training forecasts,
        and second entry an Iris cubelist of corresponding training observations.
    """
    fcs_test = xr.open_dataarray(input_path / "ESSD_benchmark_test_data_forecasts.nc")
    obs_test = xr.open_dataarray(
        input_path / "ESSD_benchmark_test_data_observations.nc"
    )

    # re-format xarray data arrays into cubes
    fcs_test.attrs["standard_name"] = "air_temperature"
    obs_test.attrs["standard_name"] = "air_temperature"
    obs_test.attrs["units"] = "K"
    fcs_test = fcs_test.sortby("station_id")
    obs_test = obs_test.sortby("station_id")
    fcs_test = fcs_test.assign_coords(
        wmo_id=("station_id", fcs_test["station_id"].values)
    )
    obs_test = obs_test.assign_coords(
        wmo_id=("station_id", fcs_test["station_id"].values)
    )
    fcs_test = fcs_test.assign_coords(station_id=np.arange(0, len(fcs_test.station_id)))
    obs_test = obs_test.assign_coords(station_id=np.arange(0, len(obs_test.station_id)))

    # rename remaining coordinates in line with Improver norms
    fcs_test = fcs_test.rename(
        {
            "step": "forecast_period",
            "number": "realization",
            "station_id": "spot_index",
            "time": "forecast_reference_time",
        }
    )
    fcs_test = fcs_test.sortby("forecast_reference_time")
    obs_test = obs_test.rename(
        {
            "step": "forecast_period",
            "station_id": "spot_index",
            "time": "forecast_reference_time",
        }
    )
    obs_test = obs_test.sortby("forecast_reference_time")

    # create cube lists of forecast test data, each cube in the list corresponds
    # with one forecast_period
    fcs_test_cube_list = []
    for i in range(0, len(fcs_test.forecast_period)):
        fcs_test_temp = fcs_test.isel(forecast_period=i)
        fcs_test_cube_temp = fcs_test_temp.to_iris()
        fcs_test_cube_temp.coord("forecast_reference_time").units = cf_units.Unit(
            fcs_test_cube_temp.coord("forecast_reference_time").units.origin,
            calendar="gregorian",
        )
        fcs_test_cube_temp.coord("station_latitude").rename("latitude")
        fcs_test_cube_temp.coord("station_longitude").rename("longitude")
        fcs_test_cube_temp.coord("spot_index").rename("spot_index")
        fcs_test_cube_temp.coord("forecast_reference_time").rename(
            "forecast_reference_time"
        )
        fcs_test_cube_temp.coord("forecast_period").rename("forecast_period")
        fcs_test_cube_temp.coord("wmo_id").rename("wmo_id")
        fcs_test_cube_list.append(fcs_test_cube_temp.copy())
    fcs_test_cube_list = iris.cube.CubeList(fcs_test_cube_list)

    # create cube lists of observation test data, each cube in the list corresponds
    # with one forecast_period
    obs_test_cube_list = []
    for i in range(0, len(obs_test.forecast_period)):
        obs_test_temp = obs_test.isel(forecast_period=i)
        obs_test_cube_temp = obs_test_temp.to_iris()
        obs_test_cube_temp.coord("forecast_reference_time").units = cf_units.Unit(
            obs_test_cube_temp.coord("forecast_reference_time").units.origin,
            calendar="gregorian",
        )
        obs_test_cube_temp.coord("latitude").rename("latitude")
        obs_test_cube_temp.coord("longitude").rename("longitude")
        obs_test_cube_temp.coord("spot_index").rename("spot_index")
        obs_test_cube_temp.coord("forecast_reference_time").rename(
            "forecast_reference_time"
        )
        obs_test_cube_temp.coord("forecast_period").rename("forecast_period")
        obs_test_cube_temp.coord("wmo_id").rename("wmo_id")
        obs_test_cube_list.append(obs_test_cube_temp.copy())
    obs_test_cube_list = iris.cube.CubeList(obs_test_cube_list)
    return [fcs_test_cube_list, obs_test_cube_list]


def apply_lapse_rate(forecast: Cube, orography_path: str,) -> Cube:
    """Apply a lapse rate adjustment to the input forecast.

    Args:
        forecast: Forecast with model_altitude and station_altitude coordinates.
        orography_path: Path to .csv file containing model orography.

    Returns:
        Forecast with lapse rate adjustment applied.
    """
    model_orography_df = pd.read_csv(orography_path)
    model_orography_df_cut = model_orography_df.loc[
        model_orography_df["station_id"].isin(forecast.coord("wmo_id").points)
    ]
    model_orography_df_cut = model_orography_df_cut.sort_values("station_id")
    alt_diff = (
        model_orography_df_cut.orography - forecast.coord("station_altitude").points
    )
    # Positive altitude difference means the model altitude is above the station
    # altitude i.e. a valley. Temperature will increase as altitude decreases.
    indices = []
    for index, coord_name in enumerate([c.name() for c in forecast.dim_coords]):
        if coord_name != "spot_index":
            indices.append(index)
    alt_diff = np.expand_dims(alt_diff, axis=indices)
    forecast.data = (
        forecast.data + np.broadcast_to(alt_diff, forecast.shape) * 6.5 / 1000
    )
    return forecast


def train_calibration(
    fcs_tr_cube_list: CubeList, obs_tr_cube_list: CubeList, thresholds: List,
) -> CubeList:
    """Train the reliability calibration using the EUPPBench dataset. Firstly,
    threshold the forecasts and observations and then construct and manipulate the
    reliability tables.

    Args:
        fcs_tr_cube_list: CubeList of forecasts in physical (realization) space.
        obs_tr_cube_list: CubeList of observations in physical space.
        thresholds: List of thresholds.

    Returns:
        Reliability table trained using the training dataset.
    """

    # apply thresholding for pre-determined thresholds
    print(f"train_calibration: Compute thresholds")
    fcs_tr_cube_pr = threshold.process(
        fcs_tr_cube_list, threshold_values=thresholds, collapse_coord="realization",
    )
    obs_tr_cube_thr = threshold.process(obs_tr_cube_list, threshold_values=thresholds)
    tr_cube_list = [fcs_tr_cube_pr, obs_tr_cube_thr]

    # create reliability calibration tables for the given forecast reference time
    # The construct_reliability_tables CLI call is configurable. The spot_index
    # is aggregated as part of the construct_reliability_tables call.
    print(f"train_calibration: Construct reliability tables")
    tr_rel_agg = construct_reliability_tables.process(
        *tr_cube_list,
        n_probability_bins=9,
        single_value_lower_limit=True,
        single_value_upper_limit=True,
        truth_attribute="truth=truth",
        aggregate_coordinates=["spot_index"],
    )
    tr_rel_agg.attributes["institution"] = "Met Office"

    print(f"train_calibration: Manipulate reliability tables")
    tr_rel_man = manipulate_reliability_table.process(tr_rel_agg)

    return tr_rel_man


def apply_calibration(
    rel_table_cube_list: Optional[CubeList],
    fcs_test_cube: Cube,
    thresholds: List,
    no_of_percentiles: int,
) -> List[Cube]:
    """
    Apply the reliability table to the test dataset to calibrate the forecast.

    Args:
        rel_table_cube_list: Reliability table to apply.
        fcs_test_cube: Test dataset forecasts.
        thresholds: Thresholds.
        no_of_percentiles: The number of percentiles to be generated.

    Returns:
        - Thresholded test dataset forecasts
        - Calibrated test dataset forecasts in probability space
        - Calibrated test dataset forecasts in realization space
    """
    fcs_test_pr_cube_list_uncal = []
    fcs_test_pr_cube_list_cal = []
    fcs_test_perc_cube_list_cal = []

    print(f"apply_calibration: Compute thresholds")
    # apply thresholding for pre-determined thresholds
    fcs_test_cube_pr = threshold.process(
        fcs_test_cube, threshold_values=thresholds, collapse_coord="realization",
    )
    fcs_test_pr_cube_list_uncal.append(fcs_test_cube_pr)
    if rel_table_cube_list is not None:
        # apply reliability table created from training data to calibrate test
        # forecast cube
        print(f"apply_calibration: apply_calibration")
        fcs_test_cube_cal = apply_reliability_calibration.process(
            forecast=fcs_test_cube_pr,
            reliability_table=iris.cube.CubeList(rel_table_cube_list),
        )
    else:
        fcs_test_cube_cal = fcs_test_cube

    fcs_test_pr_cube_list_cal.append(fcs_test_cube_cal)

    # convert to calibrated forecasts to percentiles
    print(f"apply_calibration: generate realizations")
    fcs_test_cube_cal_perc = generate_realizations.process(
        cube=fcs_test_cube_cal, realizations_count=no_of_percentiles
    )
    fcs_test_perc_cube_list_cal.append(fcs_test_cube_cal_perc)

    return [
        fcs_test_pr_cube_list_uncal,
        fcs_test_pr_cube_list_cal,
        fcs_test_perc_cube_list_cal,
    ]


def bias_correction(
    fcs_tr_cube_filt: Cube,
    obs_tr_cube_filt: Cube,
    fcs_test_cube_filt: Cube,
    global_bias_correction: bool = True,
) -> Tuple[Cube, Cube]:
    """
    Apply a simple bias correction to the training and test forecasts.

    Args:
        fcs_tr_cube_filt: Cube of training forecast data.
        obs_tr_cube_filt: Cube of training observation data.
        fcs_test_cube_filt: Cube of test forecast data.
        global_bias_correction: Boolean, if false bias correction is applied
                                independently at each site, otherwise the bias is
                                aggregated and the same correction is applied globally.

    Returns:
        - bias-corrected training forecast cube
        - bias-corrected test forecast cube
    """

    if global_bias_correction:
        fcs_tr_cube_filt.coord("station_name").points = fcs_tr_cube_filt.coord(
            "station_name"
        ).points.astype(str)
        obs_tr_cube_filt.coord("station_name").points = obs_tr_cube_filt.coord(
            "station_name"
        ).points.astype(str)
        fcs_tr_subset_mean = fcs_tr_cube_filt.collapsed(
            ["realization", "forecast_reference_time", "spot_index"], iris.analysis.MEAN
        )
        obs_tr_subset_mean = obs_tr_cube_filt.collapsed(
            ["forecast_reference_time", "spot_index"], iris.analysis.MEAN
        )
        difference_mean = obs_tr_subset_mean - fcs_tr_subset_mean
        indices = list(range(len(fcs_test_cube_filt.dim_coords)))
    else:
        fcs_tr_subset_mean = fcs_tr_cube_filt.collapsed(
            "realization", iris.analysis.MEAN
        )
        difference = obs_tr_cube_filt - fcs_tr_subset_mean
        difference_mean = difference.collapsed(
            "forecast_reference_time", iris.analysis.MEAN
        )
        indices = []
        for index, coord_name in enumerate(
            [c.name() for c in fcs_test_cube_filt.dim_coords]
        ):
            if coord_name != "spot_index":
                indices.append(index)

    difference_mean_data = np.expand_dims(difference_mean.data, axis=indices)
    # Mean masked difference values with zeroes.
    if np.ma.isMaskedArray(difference_mean_data):
        difference_mean_data = difference_mean_data.filled(0.0)
    difference_mean_test_data = np.broadcast_to(
        difference_mean_data, fcs_test_cube_filt.shape
    )
    fcs_test_cube_filt.data = fcs_test_cube_filt.data + difference_mean_test_data
    difference_mean_tr_data = np.broadcast_to(
        difference_mean_data, fcs_tr_cube_filt.shape
    )
    fcs_tr_cube_filt.data = fcs_tr_cube_filt.data + difference_mean_tr_data

    return fcs_tr_cube_filt, fcs_test_cube_filt


def calibration_wrapper(
    fcs_tr_cube_list: CubeList,
    obs_tr_cube_list: CubeList,
    fcs_test_cube_list: CubeList,
    obs_test_cube_list: CubeList,
    thresholds: List,
    no_of_percentiles: int,
    month_index: int,
    leadtime: int,
    bias_correction_enabled: bool = False,
    bias_correction_only: bool = False,
) -> Tuple[Cube, Cube, Cube, Cube, Cube, Cube]:
    """
    Function to extract the relevant month and lead time from the training dataset
    and from the test dataset, construct the reliability table from the training dataset
    and apply the reliability table to the test dataset.

    Args:
        fcs_tr_cube_list: Training dataset forecasts.
        obs_tr_cube_list: Training dataset observations.
        fcs_test_cube_list: Test dataset forecasts.
        obs_test_cube_list: Test dataset observations.
        thresholds: Thresholds.
        no_of_percentiles: Number of percentiles.
        month_index: The month as a number, starting at 0.
        leadtime: The lead time in hours.
        bias_correction_enabled: Whether to apply a simple bias correction prior to
                                 reliability calibration.
        bias_correction_only: Whether to apply only a simple bias correction.

    Returns:
        - Thresholded test dataset forecasts
        - Calibrated test dataset forecasts in probability space
        - Calibrated test dataset forecasts in realization space
        - Test dataset forecasts filtered by month and lead time
        - Test dataset observations filtered by month and lead time
        - Reliability table cube
    """
    print(f"Processing month: {month_index}")
    # choose forecast_reference_time month to calibrate next
    pdt = iris.time.PartialDateTime(month=month_index + 1)

    # filter training and test cubes down by pdt
    constr = iris.Constraint(forecast_reference_time=pdt)
    constr2 = iris.Constraint(forecast_period=leadtime)

    fcs_tr_cube_filt = fcs_tr_cube_list.extract_cube(constr & constr2)
    fcs_test_cube_filt = fcs_test_cube_list.extract_cube(constr & constr2)
    obs_tr_cube_filt = obs_tr_cube_list.extract_cube(constr & constr2)
    obs_test_cube_filt = obs_test_cube_list.extract_cube(constr & constr2)

    if bias_correction_enabled:
        print(f"bias_correction for month: {month_index}, lead time: {leadtime}")
        t0 = time.time()
        fcs_tr_cube_filt, fcs_test_cube_filt = bias_correction(
            fcs_tr_cube_filt, obs_tr_cube_filt, fcs_test_cube_filt, False
        )
        t1 = time.time()
        print("Time taken for bias correction: ", t1 - t0)

    if bias_correction_only:
        rel_table_cube = None
    else:
        print(f"train_calibration for month: {month_index}, lead time: {leadtime}")
        # train reliability calibration
        t0 = time.time()
        rel_table_cube = train_calibration(
            fcs_tr_cube_filt, obs_tr_cube_filt, thresholds=thresholds,
        )
        t1 = time.time()
        print("Time taken for training: ", t1 - t0)

    # apply reliability calibration
    print(f"apply_calibration for month: {month_index}, lead time {leadtime}")
    t0 = time.time()
    (
        fcs_test_pr_cube_uncal_temp,
        fcs_test_pr_cube_cal_temp,
        fcs_test_perc_cube_cal_temp,
    ) = apply_calibration(
        rel_table_cube,
        fcs_test_cube_filt,
        thresholds=thresholds,
        no_of_percentiles=no_of_percentiles,
    )
    t1 = time.time()
    print("Time taken for application: ", t1 - t0)
    return (
        fcs_test_pr_cube_uncal_temp,
        fcs_test_pr_cube_cal_temp,
        fcs_test_perc_cube_cal_temp,
        fcs_test_cube_filt,
        obs_test_cube_filt,
        rel_table_cube,
    )


def organise_and_save(
    fcs_test_pr_cube_uncal: Cube,
    fcs_test_pr_cube_cal: Cube,
    fcs_test_perc_cube_cal: Cube,
    fcs_test_cube_filt: Cube,
    obs_test_cube_filt: Cube,
    rel_table_cube: Cube,
    output_path: Path,
    month: int,
    leadtime: int,
):
    """Organise the calibration outputs and save.

    Args:
        fcs_test_pr_cube_uncal: Thresholded test dataset forecasts.
        fcs_test_pr_cube_cal: Calibrated test dataset forecasts in probability space.
        fcs_test_perc_cube_cal: Calibrated test dataset forecasts in realization space.
        fcs_test_cube_filt: Test dataset forecasts.
        obs_test_cube_filt: Test dataset observations.
        rel_table_cube: Reliability table cube.
        output_path: Output directory.
        month: The month as a number, starting at 0.
        leadtime: The lead time in hours.
    """

    # convert from iris cubes to xarray arrays
    test_uncalibrated_threshold_forecasts = xr.DataArray.from_iris(
        fcs_test_pr_cube_uncal[0]
    )
    test_calibrated_threshold_forecasts = xr.DataArray.from_iris(
        fcs_test_pr_cube_cal[0]
    )
    test_uncalibrated_realization_forecasts = xr.DataArray.from_iris(fcs_test_cube_filt)
    test_calibrated_realization_forecasts = xr.DataArray.from_iris(
        fcs_test_perc_cube_cal[0]
    )
    test_observations = xr.DataArray.from_iris(obs_test_cube_filt)

    # rename relevant coordinates to match the EUPPBench dataset
    test_calibrated_realization_forecasts = test_calibrated_realization_forecasts.assign_coords(
        spot_index=test_calibrated_realization_forecasts["wmo_id"]
    ).rename(
        {
            "spot_index": "station_id",
            "forecast_period": "step",
            "forecast_reference_time": "time",
            "realization": "number",
        }
    )

    test_uncalibrated_realization_forecasts = test_uncalibrated_realization_forecasts.assign_coords(
        spot_index=test_uncalibrated_realization_forecasts["wmo_id"]
    ).rename(
        {
            "spot_index": "station_id",
            "forecast_period": "step",
            "forecast_reference_time": "time",
            "realization": "number",
        }
    )

    test_observations = test_observations.assign_coords(
        spot_index=test_observations["wmo_id"]
    ).rename(
        {
            "spot_index": "station_id",
            "forecast_period": "step",
            "forecast_reference_time": "time",
        }
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%MZ")

    # save final arrays
    # Create output path
    output_path.mkdir(parents=True, exist_ok=True)
    if rel_table_cube is not None:
        iris.save(
            rel_table_cube,
            output_path
            / f"{timestamp}_reliability_calibration_table_month{month}_leadtime{leadtime}.nc",
        )

    test_uncalibrated_threshold_forecasts.to_netcdf(
        output_path
        / f"{timestamp}_test_uncalibrated_threshold_forecasts_month{month}_leadtime{leadtime}.nc"
    )
    test_calibrated_threshold_forecasts.to_netcdf(
        output_path
        / f"{timestamp}_test_calibrated_threshold_forecasts_month{month}_leadtime{leadtime}.nc"
    )
    test_uncalibrated_realization_forecasts.to_netcdf(
        output_path
        / f"{timestamp}_test_uncalibrated_realization_forecasts_month{month}_leadtime{leadtime}.nc"
    )
    test_calibrated_realization_forecasts.to_netcdf(
        output_path
        / f"{timestamp}_test_calibrated_realization_forecasts_month{month}_leadtime{leadtime}.nc"
    )
    test_observations.to_netcdf(
        output_path
        / f"{timestamp}_test_observations_month{month}_leadtime{leadtime}.nc"
    )


def iterate_by_leadtime(
    fcs_tr_cube_list: CubeList,
    obs_tr_cube_list: CubeList,
    fcs_test_cube_list: CubeList,
    obs_test_cube_list: CubeList,
    output_path: Path,
    thresholds: List,
    no_of_percentiles: int,
    month: int,
    leadtime: int,
    bias_correction_enabled: bool = False,
    bias_correction_only: bool = False,
):
    """
    Function to enable calibrating forecasts and saving the output when iterating over
    month and leadtime.

    Args:
        fcs_tr_cube_list: Training dataset forecasts.
        obs_tr_cube_list: Training dataset observations.
        fcs_test_cube_list: Test dataset forecasts.
        obs_test_cube_list: Test dataset observations.
        output_path: Output directory.
        thresholds: Thresholds.
        no_of_percentiles: Number of percentiles.
        month: The month as a number, starting at 0.
        leadtime: The lead time in hours.
        bias_correction_enabled: Whether to apply a simple bias correction prior to
                                 reliability calibration.
        bias_correction_only: Whether to apply only a simple bias correction.

    """
    t1 = time.time()
    (
        fcs_test_pr_cube_uncal,
        fcs_test_pr_cube_cal,
        fcs_test_perc_cube_cal,
        fcs_test_cube_out,
        obs_test_cube_out,
        rel_table_cube,
    ) = calibration_wrapper(
        fcs_tr_cube_list,
        obs_tr_cube_list,
        fcs_test_cube_list,
        obs_test_cube_list,
        thresholds,
        no_of_percentiles,
        month,
        leadtime,
        bias_correction_enabled=bias_correction_enabled,
        bias_correction_only=bias_correction_only,
    )

    t2 = time.time()
    print("Time taken: ", t2 - t1)

    organise_and_save(
        fcs_test_pr_cube_uncal,
        fcs_test_pr_cube_cal,
        fcs_test_perc_cube_cal,
        fcs_test_cube_out,
        obs_test_cube_out,
        rel_table_cube,
        output_path,
        month,
        leadtime,
    )


def main():
    """ Function to load the EUPPBench training and test data, apply a lapse rate
    correction and simple bias correction, then train and apply reliability calibration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=Path, help="Path to the input files.",
    )
    parser.add_argument(
        "output_path", type=Path, help="Path to the output files.",
    )
    parser.add_argument(
        "month",
        type=int,
        help="Month as an index e.g. January is 0, February is 1, etc",
    )
    parser.add_argument(
        "--leadtime", type=int, help="Lead time in hours.",
    )
    parser.add_argument(
        "--leadtime_batches", type=int, help="Total number of lead time batches.",
    )
    parser.add_argument(
        "--leadtime_batch_index", type=int, help="Lead time batch index.",
    )
    parser.add_argument(
        "--lower_threshold_bound",
        type=np.float32,
        default=223.15,
        help="The lower threshold bound.",
    )
    parser.add_argument(
        "--upper_threshold_bound",
        type=np.float32,
        default=313.2,
        help="The upper threshold bound.",
    )
    parser.add_argument(
        "--threshold_increment",
        type=np.float32,
        default=1,
        help="The threshold increment.",
    )
    parser.add_argument(
        "--no_of_percentiles", type=int, default=51, help="The number of percentiles.",
    )

    args = parser.parse_args()

    t0 = time.time()
    fcs_tr_cube_list, obs_tr_cube_list = get_train_forecasts_and_observations(
        args.input_path,
    )
    fcs_test_cube_list, obs_test_cube_list = get_test_forecasts_and_observations(
        args.input_path,
    )

    # apply lapse rate correction to training and test forecasts
    fcs_tr_cube_list_lapse = iris.cube.CubeList()
    fcs_test_cube_list_lapse = iris.cube.CubeList()

    for i in range(len(fcs_tr_cube_list)):
        fcs_tr_cube_list_lapse.append(
            apply_lapse_rate(
                fcs_tr_cube_list[i], args.input_path / "model_orography_on_stations.csv"
            )
        )
        fcs_test_cube_list_lapse.append(
            apply_lapse_rate(
                fcs_test_cube_list[i],
                args.input_path / "model_orography_on_stations.csv",
            )
        )

    # thresholds in kelvin
    thresholds = list(
        np.arange(
            args.lower_threshold_bound,
            args.upper_threshold_bound,
            args.threshold_increment,
        )
    )

    t1 = time.time()
    print("Time taken: ", t1 - t0)

    if args.leadtime:
        leadtimes = [args.leadtime]
    else:
        fps = [c.coord("forecast_period").points for c in fcs_test_cube_list]
        leadtime_batches = np.array_split(fps, args.leadtime_batches)
        leadtimes = leadtime_batches[args.leadtime_batch_index]

    print("Processing leadtimes = ", leadtimes)

    for leadtime in leadtimes:
        print("Processing leadtime = ", int(leadtime))
        iterate_by_leadtime(
            fcs_tr_cube_list,
            obs_tr_cube_list,
            fcs_test_cube_list,
            obs_test_cube_list,
            args.output_path,
            thresholds,
            args.no_of_percentiles,
            args.month,
            int(leadtime),
            True,
            False,
        )

    print("Done")


if __name__ == "__main__":
    main()
