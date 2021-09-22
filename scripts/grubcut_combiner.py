#combiner
from __future__ import print_function
import numpy as np
import cv2
import configparser
import os
import json
import sys
#uncomment to test on colab
#sys.path.append('/content/gdrive/MyDrive/videos_background_replacement/scripts') 
import bgsubtractor
# Find OpenCV version
(MAJOR_VER, MINOR_VER, SUBMINOR_VER) = (cv2.__version__).split('.')

def Combiner(path, bg_frame, save_dir,grubcut_iter,grubcut_ROI):
  '''
	Substitute background in videos using GrabCut algorithm
	
'''

  # Get video name from full path
  file_name = path.split('/')[-1].split('.')[0]
	# Open the video and check if it successfully done
  cap = cv2.VideoCapture(path)
  if not cap.isOpened:
    print('Unable to open: ' + path)
    exit(0)

	# Get fps and shape of the video
  if int(MAJOR_VER) < 3:
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  else:
    fps = cap.get(cv2.CAP_PROP_FPS)
  #print(fps)
  size = (int(cap.get(3)),int(cap.get(4)))

	# Create video writer
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(save_dir + file_name + '_new_BG.avi', fourcc, fps, size)

	# Resize background frame (image) to video shape
  background = cv2.resize(bg_frame, size, interpolation=cv2.INTER_AREA)
	
	# Loop over the frames of the video
  while True:
		# Grab the current frame
	  _, frame = cap.read()
	    # If the frame could not be grabbed, then we have reached the end of the video
	  if frame is None:
	   	break
  
    # Convert it to grayscale
	  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
	    # Initial mask with zeroes
	  mask = np.zeros(frame.shape[:2], np.uint8)
	    # These are arrays used by the algorithm internally.
	  bgdModel = np.zeros((1, 65), np.float64) #Temporary array for the background model
	  fgdModel = np.zeros((1, 65), np.float64) #Temporary arrays for the foreground model 
        # Specify a region of interest (RoI) and apply grabCut algorithm. ROI containing a segmented object. 

	    # Number of iterations the algorithm should run is 1
	    # which is fast but not good for correct segmentation  
	  cv2.grabCut(frame, mask, grubcut_ROI, bgdModel, fgdModel, grubcut_iter, cv2.GC_INIT_WITH_RECT)
	    # New mask for moving object
	  mask2 = np.where((mask == 2) | (mask == 0), (0,), (1,)).astype('uint8')
	  frame = frame * mask2[:, :, np.newaxis]
	  mask_1 = frame > 0
	  mask_2 = frame <= 0
	    # Linear combination of bgd and fgd frames with mask_1 and mask_2 "scaliars"
	  combination = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) * mask_1 + background * mask_2
	  combination = combination.astype(dtype=np.uint8)
	    # Save combined frame
	  out.write(combination)

	# When everything done, cleanup the camera and close any open windows
  print('Background substitution is successfully finished')
  cap.release()
  cv2.destroyAllWindows()


def main():

# Read config file
	
  with open('./scripts/config.json', 'r') as config_file:
    config = json.load(config_file)
	
# Parameters: get necessary information from config file
  INPUT_PATH = config['paths']['INPUT_PATH']
  OUTPUT_PATH = config['paths']['OUTPUT_PATH']
  BG_VIDEO_NAME = config['background']['backg_video_name']
  BACKGROUND_PATH = config['paths']['BACKGROUND_PATH']
  background_path=BACKGROUND_PATH + BG_VIDEO_NAME
  #extract the background from the given video
  nb_frames=config['nb_frames']['number_frames']
  bg_algo=config['bg_algo']['bg_algo'] #option in config.json: median,gmm
  grubcut_iter=config['grubcut_iter']['grubcut_iter']
  grubcut_ROI=config['grubcut_ROI']['grubcut_ROI']
  
  #Extract the background
  background = bgsubtractor.extract_background(background_path,nb_frames,bg_algo)
  
  for file in os.listdir(INPUT_PATH):
    if file.endswith('.avi') or file.endswith('.mp4'):
      path=os.path.join(INPUT_PATH, file)
      print("Precessing.................:",path)
      Combiner(path, background, OUTPUT_PATH, grubcut_iter,grubcut_ROI)

if __name__ == '__main__':
    main()
