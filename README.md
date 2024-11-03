# OPTICAL MTF

The program `optical mtf` provides a measurement technique to capture imaging quailty of electro-optical and optical systems used in physics, space technologies and industry.
It is based on quantifying the optical modulation transfer function (MTF) through a series of physical and mathematical operations which is called slanted edge method.

In the slanted edge method, a knife edge image target is used and a portion of it is taken for analysis. This portion of image is called Region of Interest (ROI). Pixels in ROI are projected onto a new axis by multiplying trigonometric functions including slanted edge's angle (oversampling).
By averaging and binning of pixel values on the new axis or curve fitting methods, an Edge Spread Function, which is the response of the optical system to a knife edge,  is obtained. Thereafter, a numerical derivation and Fourier Transform are applied to obtain MTF graph.

## Requirements

### Install requirements using pip

To install the required packages using pip, use

```sh
pip install -r requirements.txt
```
It is recommended to install in a virtual python environment.

If you're using the system wide installed version of python, you might consider the
``--user`` option. 


## Usage

To run ESF curve fitting on a sample image:
```sh
python ESF_Curve_Fitting.py
```
Once this code snippet is run, a black-white image with a knife edge rotated 12.5 degree on clockwise (Sample_image.png) will appear with a left mouse-clicking option.
Approaching the border of passing black to white region(on edge) and  left-clicking  on an edge point determines the ROI. After that, pressing space bar 2 times respectively will lead user to following calculations and final plots of Edge Spread Function, Line Spread Function and Modulation Transfer Function.


`esfcurvefitting_8.py` file contains the functions of Edge Spreaf Function(ESF), Line Spread Function(LSF) and Modulation Tranfer Function (MTF) calculations.

### To run the GUI:
```sh
python GUI.py
```
### GUI manual:
- Step 1 - Click "start button" to start camera capturing. 
- Step 2 - Click "pick sample" button to enable ROI (region of interest) selection using cursor. 
- Step 3 - Left-click on an edge that you want to perform the calculations. Selected ROI image is displayed in a pop-up windows. Press 'q' key to close the window.
- Step 4 - Live image capturing, calculations and plotting results will be displayed on the GUI.
