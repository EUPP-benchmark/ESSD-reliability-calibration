# ESSD-reliability-calibration

Reliability Calibration (RC) scripts for the ESSD benchmark. The methodology and technical details for Reliability Calibration are provided below.

This code is provided as supplementary material with:

* Demaeyer, J., Bhend, J., Lerch, S., Primo, C., Van Schaeybroeck, B., Atencia, A., Ben Bouallègue, Z., Chen, J., Dabernig, M., Evans, G., Faganeli Pucer, J., Hooper, B., Horat, N., Jobst, D., Merše, J., Mlakar, P., Möller, A., Mestre, O., Taillardat, M., and Vannitsem, S.: The EUPPBench postprocessing benchmark dataset v1.0, Earth Syst. Sci. Data Discuss. [preprint], https://doi.org/10.5194/essd-2022-465, in review, 2023.

**Please cite this article if you use (a part of) this code for a publication.**

## Methodology

This approach for calibrating ECMWF forecasts using a 20-year reforecast dataset uses multiple complementary steps to improve the forecast skill. For the bias correction and Reliability Calibration steps described below, each month is processed separately so, for example, the calibration of January using the bias correction and Reliability Calibration steps only uses information from January within the training dataset to calibrate January within the test dataset.

### Lapse rate correction
Firstly, a lapse rate correction of 6.5 K/Km between the station altitude and the model orography is applied. This matches the correction made to the uncalibrated forecasts prior to verification and provides a significant initial improvement to the forecast bias.

### Bias correction
To further improve the forecast bias, a deterministic bias correction is computed. This involves finding the difference between the ensemble mean and the observations for each site and each lead time within the training dataset. This difference is added to the forecasts within the training dataset and the test dataset as an additive bias correction.

### Reliability Calibration
The Reliability Calibration approach largely follows Flowerdew (2014), however, a more up-to-date paper is in preparation to better capture the details of the approach used within the IMPROVER codebase. Reliability Calibration specifically targets improving the reliability penalty component of the forecast whilst preserving the resolution component. Reliability Calibration directly calibrates probabilistic forecasts, which can be advantageous if the intended output is a probabilistic forecast. In this study, probabilistic forecasts are created by thresholding the input ensemble member forecasts at 0.5 K temperature intervals between 223.15 K and 313.2 K. To calibrate these probabilistic forecasts using Reliability Calibration, the number of probability bins needs to be specified. Choosing a smaller number of bins aids ensuring a sufficient sample size within each bin, whilst choosing a larger number of bins allows the calibration to be more flexible. Following Flowerdew (2014), the reliability diagrams are assumed to be near-linear across the bulk of the probability range, so 9 bins are chosen in this study. Of these 9 bins, two bins are selected to be single value bins at 0 and 1 with the remaining 7 bins being equally spaced across the probability range.

Using the IMPROVER codebase (https://github.com/metoppv/improver), reliability tables at each threshold are constructed for each site prior to aggregation into a single reliability table valid for all sites. This aggregation helps ensure adequate sample size, particularly at relatively poorly populated thresholds, however this is at the possible detriment of combining information from sites that are not similar in terms of their reliability penalties. After aggregation, the reliability table is further “manipulated” to ensure that it is suitable for calibration purposes. These manipulations include:

1.	Combining low sample count bins with their neighbour.
2.	If there is non-monotonicity with observation frequency, combining an additional bin pair.
3.	If non-monotonicity with observation frequency remains, assuming a constant observation frequency for any portion of the reliability diagram that is still non-monotonic.

This procedure will be described in more detail in a separate publication.

The reliability tables are then applied to each forecast for calibration. The input forecast probability for the current forecast is replaced by the observation frequency given by linearly interpolation. This ensures that even though the reliability table has a discrete number of bins of forecast probability, any value of observation frequency can potentially be produced. As the exceedance probabilities at each threshold are calibrated separately, the calibrated probabilities could be non-monotonic as a function of threshold. As a final step, monotonicity of the calibrated probabilities across thresholds is enforced.

#### Reference
Flowerdew, J., 2014: Calibrating ensemble reliability whilst preserving spatial structure. Tellus A Dyn. Meteorol. Oceanogr., 66, 22662, https://doi.org/10.3402/tellusa.v66.22662.


## Technical details

First, if you do not have it, get the ESSD benchmark dataset using [the download script](https://github.com/EUPP-benchmark/ESSD-benchmark-datasets). This will fetch the dataset into NetCDF files on your disk.

Then, to run the code, first clone the repository:

```
git clone https://github.com/EUPP-benchmark/ESSD-reliability-calibration
```

Then create a conda environment:

```
conda env create -f environment.yml
conda activate ESSD-reliability-calibration
```

The python scripts should now be runnable.

Prior to the calibration steps, the code includes functions to manipulate the input dataset into a dataset that is more inline with the [CF Conventions](https://cfconventions.org/), and therefore compatible with the [iris](https://scitools-iris.readthedocs.io/en/latest/) and [improver](https://improver.readthedocs.io/en/latest/) modules.

The script provided can be run as follows to run each month and lead time independently:
```
python ./ eumetnet_improver_application_per_month.py input_path output_path 0 --leadtime=6
```

Alternatively, multiple lead times can be run within a single call by grouping the lead times into a specified number of batches and selecting to run a specific batch:
```
python ./ eumetnet_improver_application_per_month.py input_path output_path 0 --leadtime_batches=11 --leadtime_batch_index=0
```

After running the eumetnet_improver_application_per_month.py script for each month and lead time of interest, the output from these scripts then need to be joined together.

Firstly all lead times are joined together for each month:
```
python ./eumetnet_join_leadtimes.py input_path output_path
```

Then each month is joined together:
```
python ./eumetnet_join_months.py input_path output_path
```

### Runtime
Each month and lead time combination takes approximately 40 minutes using < 4GB of RAM. The vast amount of this processing time is the Reliability Calibration training with the Reliability Calibration application taking ~30 seconds.

