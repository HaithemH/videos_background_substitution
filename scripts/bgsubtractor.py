from __future__ import print_function
import numpy as np
import cv2
import configparser
import json
from sklearn.mixture import GaussianMixture

def extract_background(background_video,nb_frames,bg_algo):
	
  #print(background_video)
  cap = cv2.VideoCapture(background_video)
  
  if not cap.isOpened:
	    print('Unable to open: ' + BACKGROUND_PATH + BG_VIDEO_NAME)
	    exit(0)

	# Randomly select 25 frames
  frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=nb_frames)
   
	# Store selected frames in an array
  frames = []
  for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
  
  if bg_algo =='median': #it can use Gaussian Mixture Models
	# Calculate the median along the time axis
    background = np.median(frames, axis=0).astype(dtype=np.uint8)
  else:
 
#model each point in space for all the three image channels, 
#namely R, G and B as a bimodal distribution of Gaussians,
#where one Gaussian in the mixture accounts for the background and the other for the foreground.
    gmm = GaussianMixture(n_components = 2)
   # initialize a dummy background image with all zeros
    frames = np.array(frames)
    background = np.zeros(shape=(frames.shape[1:]))
    for i in range(frames.shape[1]):
      for j in range(frames.shape[2]):
        for k in range(frames.shape[3]):
            X = frames[:, i, j, k]
            X = X.reshape(X.shape[0], 1)
            gmm.fit(X)
            means = gmm.means_
            covars = gmm.covariances_
            weights = gmm.weights_
            idx = np.argmax(weights)
            background[i][j][k] = int(means[idx])
	# Save median frame to output folder
    cv2.imwrite(OUTPUT_PATH + 'background.jpg', background)

	
	# When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()
  
  return background
  
if __name__ == '__main__':


  # Read json config file
  with open('./scripts/config.json', 'r') as config_file:
    config = json.load(config_file)
	
	# Get necessary information from config file
  INPUT_PATH = config['paths']['INPUT_PATH']
  OUTPUT_PATH = config['paths']['OUTPUT_PATH']
  BACKGROUND_PATH = config['paths']['BACKGROUND_PATH']
  BG_VIDEO_NAME = config['background']['backg_video_name']
  nb_frames=config['nb_frames']['number_frames']
  bg_algo=config['bg_algo']['bg_algo'] #option in config.json: median,gmm

  background_path=BACKGROUND_PATH + BG_VIDEO_NAME
  extract_background(background_path,int(nb_frames),bg_algo)
