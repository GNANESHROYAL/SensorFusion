# Enhanced Sensor Fusion (ESF) for Social-Aware Robot Navigation  
LiDAR ✕ RGB -- better human-aware path-planning
---

> **Paper:** *Enhancing Social-Aware Autonomous Navigation Scheme with LiDAR and RGB Sensor Fusion*  

This repository hosts the code, simulation assets, and supplementary material for the **ESF** model—an early-fusion pipeline that projects 2-D LiDAR points onto RGB frames, filters them with YOLOv8 detections, and feeds the fused features into an RNN-Transformer stack for real-time decision-making.  
Compared with DR-SPAAM, ESF cuts inference time by **≈ 48 %** while doubling the frame rate, with comparable recall and F1-score 
---

## Quick Start (Ubuntu 22.04 + ROS 2 Humble)

```bash
# clone repo
git clone https://github.com/<user>/sensor-fusion-esf.git
cd sensor-fusion-esf

# create Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # ultralytics, opencv-python, numpy, etc.

# build ROS workspace
colcon build --symlink-install
source install/setup.bash

# launch Gazebo demo (robot + 5 humans)
ros2 launch esf_demo gazebo_social_world.launch.py
