# Prerequisites

Python 3.12


# Installation

1. Download the file at https://www.datalumos.org/datalumos/project/240591/version/V1/view?path=/datalumos/240591/fcr:versions/V1/transmission-lines-1-shapefile.zip&type=file and unzip the large csv-file to *data/raw/Transmission_Lines*.

2. Go to https://solargis.com/resources/free-maps-and-gis-data?locality=usa, download the file called *LTAy Yearly Monthly Totals GEOTIFF*, and unzip *western-hemisphere/PVOUT.tif* to *data/raw*.


# Usage
We will use the included model *CA_NV_AZ_UT_config* in *configs.py* here as an example; this is a model for the territory comprising California, Nevada, Arizona and Utah. *configs.py* contains all the parameters you may want to adjust to your needs.

* *(Optional)* Before modeling the market, we need to choose the territory to explore and approximate its geometry with a mesh. You can play around with mesh parameters *h_km* (mesh size) and *simplify_km* (boundary simplification size) and explore figures and diagnostics:
```
python run_mesh_diag.py --h_km 6 --simplify_km 18 --states '[CA,NV,AZ,UT]'
```
Typically *simplify_km* of 3 x *h_km* works well.


* Fit a model defined in *configs.py*. For example, for the included model *CA_NV_AZ_UT_config*:
```
python run_fit.py --model CA_NV_AZ_UT_config
```
A number of figures will be saved in *out/CA_NV_AZ_UT_config/figures*, and a json file with the model parameters will appear in *out/*.


* Print the metrics and generate diagnostic plots:
```
python run_metrics.py --model CA_NV_AZ_UT_config
```
In addition to the outputs on the screen, a few plots will be saved to *out/CA_NV_AZ_UT_config/metrics*.


* Run the simulations:
```
python run_simulations.py --model CA_NV_AZ_UT_config
```
In addition to the data on the screen, a few plots and animations will appear in *out/simulations*.
