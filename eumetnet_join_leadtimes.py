"""
Join outputs produced for separate lead times into a separate files for each month.
"""

import argparse
import time
from pathlib import Path

import xarray as xr


def load_merge_save(
    input_path: Path, output_path: Path, input_pattern: str, output_filename: str
):
    """Load all files and merge into a single xarray data array. Correct attributes
    and save to netCDF.

    Args:
        input_path: Input directory
        output_path: Output directory.
        input_pattern: Wildcarded pattern for the input files to be read.
        output_filename: Output file name.
    """

    dataarrays = []
    for afile in input_path.glob(input_pattern):
        dataarrays.append(xr.open_dataarray(input_path / afile))

    try:
        dataarrays[0].step
    except AttributeError:
        forecast_period_coord_name = "forecast_period"
    else:
        forecast_period_coord_name = "step"

    if dataarrays[0][forecast_period_coord_name].dims:
        dataarray = xr.merge(dataarrays)
    else:
        dataarray = xr.concat(dataarrays, "step")

    output_path.mkdir(parents=True, exist_ok=True)
    dataarray.to_netcdf(output_path / output_filename)


def main():
    """Merge output from different leadtimes into a single netCDF file."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=Path, help="Path to the input files.",
    )
    parser.add_argument(
        "output_path", type=Path, help="Path to the output files.",
    )
    args = parser.parse_args()

    for month in range(12):

        input_patterns = [
            f"*test_uncalibrated_realization_forecasts_month{month}_*.nc",
            f"*test_calibrated_realization_forecasts_month{month}_*.nc",
            f"*test_uncalibrated_threshold_forecasts_month{month}_*",
            f"*test_calibrated_threshold_forecasts_month{month}_*",
            f"*test_observations_month{month}_*",
        ]
        output_filenames = [
            f"test_uncalibrated_realization_forecasts_month{month}_merged.nc",
            f"test_calibrated_realization_forecasts_month{month}_merged.nc",
            f"test_uncalibrated_threshold_forecasts_month{month}_merged.nc",
            f"test_calibrated_threshold_forecasts_month{month}_merged.nc",
            f"test_observations_month{month}_merged.nc",
        ]

        for input_pattern, output_filename in zip(input_patterns, output_filenames):
            print(f"Processing {input_pattern}")
            t0 = time.time()
            load_merge_save(
                args.input_path,
                args.output_path,
                input_pattern=input_pattern,
                output_filename=output_filename,
            )
            t1 = time.time()
            print("Time taken: ", t1 - t0)


if __name__ == "__main__":

    main()
