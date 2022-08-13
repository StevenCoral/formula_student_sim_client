# Formula Racing Simulation for Ben Gurion University
## Introduction
This repository holds an Airsim-based client for simulating the FSAE challenge within Unreal Engine.  
It was created in 2021, hence is relevant for the challenge made in that year.  
The scripts were tested on Windows 10, using Python 3.9, Airsim 1.5.0 and Unreal Engine 4.26.2.  
Testing on newer versions of Airsim produces an error in the simGetImages function, which may be solvable.  
For (much) more details, please review the project PDF file included in the repo.  

## Preparation
In order to install and use this project, you should:  
* Install Unreal Engine using formal instructions.
* Download Unreal Project (zip file) from the [following link](https://drive.google.com/drive/folders/1_BdXtkc-P8FzvNqy38et9genC3dfL_1K?usp=sharing "project files location").
* Install Airsim using formal instructions (binaries should already exist under the project plugins).
* Recommended: create a Python Virtual Environment for this project (Venv / Anaconda / else).
* Type "pip install wheel".
* pip-install the rest of the dependencies within the provided "requirements.txt" file.
* Recommended: create a Pycharm project in the repo folder.
* Mark the "utils" subfolder as Sources Root or add it to Python path in-code using sys.
* Copy "settins.json" into <Documents>/Airsim folder, backup and overwrite if needed.
* Open StudentRacing project in UE 4.26 and make sure Airsim plugin is enabled.
* Click on "Play" and **wait for all shaders to compile**.
* Play the project and then run "main_program.py" script.

## Usage
The main program is made of 2 main missions:  
1. Discovering cones using LIDAR and classifying them using the cameras and saving their location.  
2. Generating a path spline from the positions of the discovered cones.  
3. Closing a control loop over a spline, given said spline.  
  
There is some hardcoded ability to log data that was crucial for exporting statistics, but this will probably not mean much to you.  
Feel free to utilize the mechanism to your own needs if applicable.  
Theoretically, the UE cone course can be changed into whatever, but changing the vehicle starting point might cause issues due to the way Airsim works.  
The sub-missions presented above can be used, tested or tweaked separately, if provided with the right parameters.  
As mentioned, see project PDF file for more info.  






