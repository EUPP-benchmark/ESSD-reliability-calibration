"""Join outputs produced for separate months."""

import argparse
from pathlib import Path

import xarray as xr


def load_merge_save(
    input_path: Path, output_path: Path, input_pattern: str, output_filename: str
):
    """Load all files and merge into a single xarray data array. Correct attributes
    and save to netCDF.

    Args:
        input_path: Input directory
        output_dir: Output directory
        input_pattern: Wildcarded pattern for the input files to be read.
        output_filename: Output file name.
    """

    dataarrays = []
    for afile in sorted(input_path.glob(input_pattern)):
        if "leadtime" in str(afile.stem):
            continue
        print(f"Processing {afile}")
        da = xr.open_dataarray(input_path / afile)
        try:
            da.forecast_period
        except AttributeError:
            forecast_period_coord_name = "step"
        else:
            da = da.swap_dims({"step": "forecast_period"})
            forecast_period_coord_name = "forecast_period"
        da = da.drop_duplicates(forecast_period_coord_name)
        da = da.sortby(forecast_period_coord_name)
        if forecast_period_coord_name == "forecast_period":
            da = da.swap_dims({"forecast_period": "step"})
        dataarrays.append(da)

    dataarray = xr.merge(dataarrays)

    if dataarray.get("air_temperature", None) is not None:
        dataarray = dataarray.rename({"air_temperature": "t2m"})

    dataarray.attrs["tier"] = "1"
    dataarray.attrs["experiment"] = "ESSD-benchmark"
    dataarray.attrs["institution"] = "MetOffice"
    dataarray.attrs["model"] = "IMPROVER-reliabilitycalibration-v1.3.1"
    dataarray.attrs["version"] = "v1.3"
    dataarray.attrs["output"] = "quantiles"

    if dataarray.get("t2m", None) is not None:
        dataarray.t2m.attrs["tier"] = "1"
        dataarray.t2m.attrs["experiment"] = "ESSD-benchmark"
        dataarray.t2m.attrs["institution"] = "MetOffice"
        dataarray.t2m.attrs["model"] = "IMPROVER-reliabilitycalibration-v1.3.1"
        dataarray.t2m.attrs["version"] = "v1.3"
        dataarray.t2m.attrs["output"] = "quantiles"

    output_path.mkdir(parents=True, exist_ok=True)
    dataarray.to_netcdf(output_path / output_filename)


def main():
    """Merge output from different months into a single netCDF file."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=Path, help="Path to the input files.",
    )
    parser.add_argument(
        "output_path", type=Path, help="Path to the output files.",
    )
    args = parser.parse_args()

    input_patterns = [
        "test_uncalibrated_realization_forecasts_month*merged.nc",
        "test_calibrated_realization_forecasts_month*_merged.nc",
        "test_uncalibrated_threshold_forecasts_month*_merged.nc",
        "test_calibrated_threshold_forecasts_month*_merged.nc",
        "test_observations_month*_merged.nc",
    ]

    output_filenames = [
        "test_uncalibrated_realization_forecasts_merged.nc",
        "test_calibrated_realization_forecasts_merged.nc",
        "test_uncalibrated_threshold_forecasts_merged.nc",
        "test_calibrated_threshold_forecasts_merged.nc",
        "test_observations_merged.nc",
    ]

    for input_pattern, output_filename in zip(input_patterns, output_filenames):
        print(f"Processing {input_pattern}")
        load_merge_save(
            args.input_path,
            args.output_path,
            input_pattern=input_pattern,
            output_filename=output_filename,
        )


if __name__ == "__main__":

    main()
