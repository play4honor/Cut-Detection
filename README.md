### Segment NFL game film into All-22/Endzone footage.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E_Ze1C54FzLAliopoIhTrUF-FujL_tmX)

#### Setup

Likely we need to `pip install -e .` in the project directory so `frameID` is available to import. I don't know a better way off the top of my head.

Example usage:

`python segment_video.py "video/2021.12.05_Steelers_Ravens.mp4" "steelers-ravens-segments.csv"`

#### Organization

 - `training_scripts` contains scripts related to preparing data and training a model.
 - `frameID` contains all the code loaded by the scripts, as well as a pre-trained model.