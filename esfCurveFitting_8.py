import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.fft import fftfreq, fft
import gc
import matplotlib
matplotlib.use("Agg")

def f1(ROI_Image):
  ROI_Image = ROI_Image.astype(np.float64)

  Kx = np.array([[-1,1],[-1,1]])
  Kx = (1/2)*Kx
  # Convolution Operation:
  Fx = cv.filter2D(ROI_Image, -1, Kx) 
  Fx = abs(Fx)

  # Kernel Definition of Derivating Matrix in y - direction:   
  Ky = np.array([[1,1],[-1,-1]])
  Ky = (1/2)*Ky

  # Convolution Operation:
  Fy = cv.filter2D(ROI_Image, -1, Ky) 
  Fy = abs(Fy)

      
  theta = math.atan(np.sum(Fy)/np.sum(Fx)) # in radians

  rows, cols = (ROI_Image.shape[0], ROI_Image.shape[1])
  Value_Matrix = np.zeros((rows*cols,2))

  # ESF Calculation Algorithm:
      
  i = 0
  for k in range(rows-1,-1,-1): 
      for m in range(0, cols):
          Value_Matrix[m + i,0] = (m*1.0  + math.tan(theta)*(k-(rows-1)))*math.cos(theta) # math.cos(theta) # Index Alignment
          Value_Matrix[m + i,1] = ROI_Image[k,m] # Intensity Alignment
      i = i + cols


  Value_Matrix_Trial_2 = np.copy(Value_Matrix)
  ind = np.lexsort((Value_Matrix_Trial_2[:,1], Value_Matrix_Trial_2[:,0])) 
  ESF_Matrix = Value_Matrix_Trial_2[ind] # Last Form of ESF is defined here.


  #############################################################################

  y = ESF_Matrix[:,1] # Intensity Values of ESF Data
  x = ESF_Matrix[:,0] # New and oversampled indices in the unit of 'pixel' from the ESF Data

  #############################################################################
  #############################################################################
      
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
  # The newly drawn curve is plotted on the ESF data in red...
  popt, pcov = curve_fit(fermi_f, x, y, maxfev=36000)
  print(popt)
      
  a_opt, b_opt, c_opt = popt # Optimal parameters returned by the curve_fit functionr
  x_model = np.linspace(min(x), max(x),100000)
  y_model = fermi_f(x_model, *popt)


  ############################################################################
  # Fitted ESF Function:
  fig1=plt.figure()
  plt.plot(x_model,y_model, linewidth = 3.6, color = 'r') # Fitted Function
  plt.title('Edge Spread Function')
  plt.xlabel('piksel(pxl)')
  plt.ylabel('Intensity')
  plt.grid()

  esf_plt = plt.gcf() # gets the current figure object
  plt.close(fig1)
  plt.cla()
  plt.clf()
  #gc.collect()
  
  ###############################################################################
  # LSF Derivation:
    # At this stage, as a result of the fitting process, the derivative of the new function with respect to the x variable is calculated..


  LSF = abs(np.gradient(y_model,x_model))
  fig2=plt.figure()
  plt.plot(x_model,LSF,linewidth = 3.2, color ='purple' )
  #plt.xlim([100, 120])
  plt.title('Line Spread Function')
  plt.xlabel('piksel(pxl)')
  plt.ylabel('Response')
  plt.grid()

  lsf_plt = plt.gcf()  # gets the current figure object
  plt.close(fig2)
  plt.cla()
  plt.clf()
  #gc.collect()
  
  ###############################################################################
  ###############################################################################
      
    # At this stage, the DFT (Discrete Fourier Transform) of the LSF (Line Spread Function) is calculated.
    # The Modulation Transfer Fucntion(MTF) was plotted by normalizing the data taken from the Fourier Transform..


  x = x_model
  N = len(x) 

  f2 = fftfreq(len(x),np.mean(np.diff(x))) ## -- >> Cycles per pixel

  x1_FFT = abs(fft(LSF)) ## Taking Fourier Transform !!

  norm_x1_FFT = x1_FFT/x1_FFT[0] # By dividing zero frequency component, normalizing is done.

  fig3=plt.figure()
  plt.plot(f2[:N//2], norm_x1_FFT[:N//2])
  plt.xlabel('spatial freq, cyc/pxl')
  plt.title('Modulation Transfer Function')
  plt.grid()
  plt.xlim([0,1])

  mtf_plt = plt.gcf()  
  plt.close(fig3)
  plt.cla()
  plt.clf()
  #gc.collect()
  
  a3 = norm_x1_FFT[:N//2] ## Normalized MTF values
  a4 = f2[:N//2] ## Frequency axis in cyc/pxl
  
  MTF_Matrix = np.zeros((a4.shape[0],2))
  MTF_Matrix[:,0] = a3
  MTF_Matrix[:,1] = a4
  
  gc.collect()
  return MTF_Matrix, esf_plt, lsf_plt, mtf_plt

  