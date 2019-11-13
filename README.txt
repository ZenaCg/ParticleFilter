## Introduction

Simultaneous Localization and Mapping (SLAM) is an important topic within the mobile robotics community. The solutions to the SLAM problem has brought possibilities for a mobile robot to be placed at an unknown location in an unknown environment and simultaneously building a map and locating itself. As it means to make robots truly autonomous, the published SLAM algorithm are widely employed in self-driving cars, autonomous underwater vehicles and unmanned aerial vehicles. In this project, within sensing data provided by encoders, an inertial measurement unit and a horizontal LIDAR scanner in hands, we design a particle filter to perform the localization methods. Furthermore, we texture the map by using the provide data from a RGBD camera.

## Data

- The data are provided by four encoders, an inertial measurement unit, a horizontal LIDAR with 270◦ degree field of view and maximum range of 30m provides distances to obstacles in the environment and a RGBD camera provides both RGB images and disparity images.

## Code

- Python script: particlefilter.py : this file contains data loading, signal processing and a particle filter.
- Python script: utils.py : this file contains helper functions for the particle filter and grid mappings.

## Results

- The results are discussed in detail within the report.
