import pixellib
from pixellib.tune_bg import alter_bg
import json
import os
def main():

# Read config file
	
  with open('./scripts/config.json', 'r') as config_file:
    config = json.load(config_file)
	
# Parameters: get necessary information from config file
  INPUT_PATH = config['paths']['INPUT_PATH']
  OUTPUT_PATH = config['paths']['OUTPUT_PATH']
  BG_VIDEO_NAME = config['background']['backg_video_name']
  MODEL = config['paths']['MODEL']
  BACKGROUND_PATH = config['paths']['MODEL']
  background_path=OUTPUT_PATH + "background.jpg"
  
  for file in os.listdir(INPUT_PATH):
    if file.endswith('.avi') or file.endswith('.mp4'):
      path=os.path.join(INPUT_PATH, file)
      print("Precessing.................:",path)
      change_bg = alter_bg(model_type="pb")
      change_bg.load_pascalvoc_model(MODEL+"xception_pascalvoc.pb")
      change_bg.change_video_bg(path, background_path, frames_per_second = 10, output_video_name=OUTPUT_PATH +'PV_'+ file, detect = "person")
 
if __name__ == '__main__':
    main()
