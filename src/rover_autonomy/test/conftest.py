import os
import sys

# Import rover_autonomy straight from the source folder, so the tests run without ROS or a build.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
