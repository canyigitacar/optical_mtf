
 # -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:35:21 2023

@author: Yiğit
"""

## Date: 11.12.2022

# Optical Imaging Operations:

# Calling Useful Libraries:
 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import os
from copy import deepcopy
from timeit import default_timer as timer

###############################################################################
# Part 1:

# Reading the first Image File:
sample_image_name = "Sample_image.png"

img_1 = cv.imread(sample_image_name,0)

###############################################################################
###############################################################################

coordinate_1 = []
coordinate_2 = []

def click_event(event, c1, c2, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(c1, ' ', c2) 
        coordinate_1.append(c1)
        coordinate_2.append(c2)
        #return c1,c2
    
        # displaying the coordinates 
        # on the image window 
        font = cv.FONT_HERSHEY_SIMPLEX 
        cv.putText(img, str(c1) + ',' +
                    str(c2), (c1,c2), font, 
                    1, (255, 0, 0), 2) 
        cv.imshow('image', img) 
        
# driver function 
if __name__=="__main__": 
  
    # reading the image 
    img = cv.imread(sample_image_name, 0) 
    # displaying the image 
    cv.imshow('image', img) 
    
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv.setMouseCallback('image', click_event) 
  
    # wait for a key to be pressed to exit 
    cv.waitKey(50000) 
  
    # close the window 
    cv.destroyAllWindows()

###############################################################################
start = timer()

# Part 2:

# Determining the ROI from the entire image:

x1 = int(np.array(coordinate_1))
y1 = int(np.array(coordinate_2))
margin = 10
ROI_Image = deepcopy(img_1[y1-margin:y1+margin, x1-margin:x1+margin])
cv.imshow("ROI_OS", ROI_Image);
cv.waitKey(80000);
cv.destroyAllWindows();


###############################################################################
    
ROI_Image = ROI_Image.astype(np.float64)

###############################################################################
###############################################################################

# Part 3:

## Taking Gradient of an Image to find Slanted Edge's Angle (theta):
    
# First of all, derivates will be taken in x and y directions.
# Derivative processing will be performed by convolving the "ROI_Image" matrix with a specially defined "kernel"
# As a result of this convolution process, new matrices will be obtained that hold the changes in the "x and y directions".

# When the elements of the matrices that hold the changes in these directions are summed and compared to each other, 
# the slanted edge angle (theta) will be obtained.
 
### Taking Derivative in x Direction:

# Kernel Definition of Derivating Matrix in x - direction:        
Kx = np.array([[-1,1],[-1,1]])
Kx = (1/2)*Kx
# Convolution Operation:
Fx = cv.filter2D(ROI_Image, -1, Kx) 
Fx = abs(Fx)

###############################################################

### Taking Derivative in y Direction:
    
# Kernel Definition of Derivating Matrix in y - direction:   
Ky = np.array([[1,1],[-1,-1]])
Ky = (1/2)*Ky

# Convolution Operation:
Fy = cv.filter2D(ROI_Image, -1, Ky) 
Fy = abs(Fy)


###############################################################

# Slanted Edge Angle (theta) Calculation:

# For the calculation in this part, see this article as reference: " https://opg.optica.org/josk/abstract.cfm?URI=josk-20-3-381 "
    
theta = math.atan(np.sum(Fy)/np.sum(Fx)) # in radians
theta_in_degrees = 180*(theta/np.pi); # in degrees

# Part 4:
   
# In this section, an ESF (Edge Spread Function) will be extracted from the discrete ROI..
# While doing this process, each line in the ROI will be scanned and an "oversampling" will be performed by multiplying the "x" variable, 
# which holds the indices of the pixel values ​​in the horizontal direction, with tan(theta) and then cos(theta).

# In other words, the "oversampled" x values ​​in the first column of the newly created 2-dimensional matrix are kept, 
# and the y values,  the "Intensity" values ​​that have not been changed, are kept in the second column.

rows, cols = (ROI_Image.shape[0], ROI_Image.shape[1]);
Value_Matrix = np.zeros((rows*cols,2))

# ESF Calculation Algorithm:
i = 0;
for k in range(rows-1,-1,-1): 
    for m in range(0, cols):
        Value_Matrix[m + i,0] = (m*1.0  + math.tan(theta)*(k-(rows-1)))*math.cos(theta) # math.cos(theta) # Index Alignment
        #print(m*1.0 + math.tan(theta)*k)
        Value_Matrix[m + i,1] = ROI_Image[k,m] # Intensity Alignment
    i = i + cols
    
    
# In the following Code Snippet, 
# the final state of ESF is reached by sorting from the two-dimensional matrix (Value Matrix) produced from smallest to largest:

Value_Matrix_Trial_2 = np.copy(Value_Matrix)
ind = np.lexsort((Value_Matrix_Trial_2[:,1], Value_Matrix_Trial_2[:,0])) 
ESF_Matrix = Value_Matrix_Trial_2[ind] # Last Form of ESF is defined here.

plt.scatter(ESF_Matrix[:,0], ESF_Matrix[:,1], c ="blue")
plt.title('ESF_TRIAL')
plt.grid()
plt.show()

###############################################################################

y = ESF_Matrix[:,1] # Intensity Values of ESF Data
x = ESF_Matrix[:,0] # New and oversampled indices in the unit of 'pixel' from the ESF Data

###############################################################################
###############################################################################

# Part 5: 
    
  # There are 2 ways after obtaining DISCRETE ESF Data:
      
   # 1.  Smoothing the ESF Data by applying 1/4 pixel sampling.
   # 2.  Curve Fitting with an appropriate function(e.g., Fermi Function) by not doing any modification to ESF Data
   

#################################################################
# Continuing with the second method:
    
    # If a Fermi function is defined as follows and fitted to the ESF data,:

# Defintion of the Fermi Function:
    
def fermi_f(x,a,b,c):
    result = a/(np.exp(-1*((x-b)/c)) + 1)
    return result

#############################################################

# At this stage, Python's "curve_fit" command was used..
# The Fermi curve has been redrawn with three new parameters returned by the Curve_Fit function..
# The newly drawn curve is plotted on the ESF data in red..

from scipy.optimize import curve_fit
popt, pcov = curve_fit(fermi_f, x, y)
print(popt)
    
a_opt, b_opt, c_opt = popt # Optimal parameters returned by the curve_fit function
x_model = np.linspace(min(x), max(x),100000)
y_model = fermi_f(x_model, *popt)

plt.figure()
plt.scatter(x,y) # ESF DATA
plt.plot(x_model,y_model, linewidth = 3.6, color = 'r') # Fitted Function
plt.title('Curve_Fitted_Edge Spread_Function_1')
plt.xlabel('piksel(pxl)')
plt.ylabel('Intensity')
plt.grid()
plt.show()

############################################################################
# Fitted ESF Function:
    
plt.figure()
plt.plot(x_model,y_model, linewidth = 3.6, color = 'r') # Fitted Function
plt.title('Edge Spread Function')
plt.xlabel('piksel(pxl)')
plt.ylabel('Intensity')
plt.grid()
plt.show()

###############################################################################
# LSF Derivation:
  # At this stage, as a result of the fitting process, the derivative of the new function with respect to the x variable is calculated..

LSF = abs(np.gradient(y_model,x_model))
plt.figure()
plt.plot(x_model,LSF,linewidth = 3.2, color ='purple' )
plt.title('Line Spread Function')
plt.xlabel('piksel(pxl)')
plt.ylabel('Response')
plt.grid()
plt.show()

###############################################################################
###############################################################################

# Part 6:
    
  # At this stage, the DFT (Discrete Fourier Transform) of the LSF (Line Spread Function) is calculated.
  # The Modulation Transfer Fucntion(MTF) was plotted by normalizing the data taken from the Fourier Transform..

from scipy.fft import fftfreq,fft  
x = x_model
N = len(x) 

f2 = fftfreq(len(x),np.mean(np.diff(x))) ## -- >> Cycles per pixel

x1_FFT = abs(fft(LSF)) ## Taking Fourier Transform !!

norm_x1_FFT = x1_FFT/x1_FFT[0] # By dividing zero frequency component, normalizing is done.

plt.figure()
plt.plot(f2[:N//2], norm_x1_FFT[:N//2])
plt.xlabel('spatial freq, cyc/pxl')
plt.title('Modulation Transfer Function')
plt.grid()
plt.xlim([0,1])
plt.show()

a3 = norm_x1_FFT[:N//2] ## Normalized MTF values
a4 = f2[:N//2] ## Frequency axis in cyc/pxl
 
MTF_Matrix = np.zeros((a4.shape[0],2))
MTF_Matrix[:,0] = a3
MTF_Matrix[:,1] = a4

directory = r"output"
filename = "mtf_main_data.txt"
 
 # Ensure the directory exists
if not os.path.exists(directory):
   os.makedirs(directory)
   
 # Step 3: Create the full file path
file_path = os.path.join(directory, filename)
 
 # Step 4: Save the array to the text file
np.savetxt(file_path, MTF_Matrix, fmt="%.8f")

print(f"Array saved to {file_path}")

end = timer()
print(end - start)

