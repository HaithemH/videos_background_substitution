from __future__ import print_function
import numpy as np
import cv2
import configparser
from os import walk
import os
import json
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
def mo_sub(video,output,mo_fg_algo):

#Function to substract moving objects from the backgroud

	# Open video and check if it successfully opened
  #print(video)
  cap = cv2.VideoCapture(video)
  if not cap.isOpened:
	  print('Unable to open: ' + video)
	  exit(0)

		# Get fps and shape of the video
  if int(major_ver)  < 3:
	  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  else:
	  fps = cap.get(cv2.CAP_PROP_FPS)
  size = (int(cap.get(3)),int(cap.get(4))) 

		# Create video writer
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(output, fourcc, fps, size)
  

	# Create background subtractor either MOG or KNN 
  if mo_fg_algo=='mog':
    backSub = cv2.createBackgroundSubtractorMOG2()
  else:
	  backSub = cv2.createBackgroundSubtractorKNN()
  
  while True:
    _, frame = cap.read()
    #print(frame)
    if frame is None:
      
      break
    fgMask = backSub.apply(frame)
    fgMask = cv2.cvtColor(fgMask,cv2.COLOR_GRAY2RGB)#convert to RGB
	    	
    out.write(fgMask) #Save to output folder

	# When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    # Read config file
  with open('config.json', 'r') as config_file:
    config = json.load(config_file)
	
	# Get necessary information from config file
  INPUT_PATH = config['paths']['INPUT_PATH']
  OUTPUT_PATH = config['paths']['OUTPUT_PATH']
  mo_fg_algo=config['mo_fg_algo']['mo_fg_algo']
  filenames = next(walk(INPUT_PATH), (None, None, []))[2]
  #print(filenames)
  for file in filenames:
		#process all file in the input directory and save to the output directory
    f=os.path.splitext(file)[0]
    video=INPUT_PATH + file
    output=OUTPUT_PATH+f+'_mo.avi'
    mo_sub(video,output,mo_fg_algo)