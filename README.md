# Visual-Inertial-Odometry-for-Agent-s-Motion-
## Introduction

The first step to be able to implement a complete Simultaneous Localization And Mapping
(SLAM) system, is the visual odometry. Visual Odometry (VO) aims to continuously estimate
the scaled pose (position and orientation) with respect to the camera body frame (c) of the
moving body (drone – vehicle – Agent) using a pipeline of image frames. Some cameras that
are used to perform this task are equipped with an Inertial Measurement Unit (IMU), which
measures the linear accelerations and angular velocities with respect to the IMU body frame (i).
The accelerations and angular velocities measured from the IMU can be integrated and we can
obtain the positions, velocities and orientations but the main problem with that integration is
that it accumulates errors coming from the biases of the IMU (accelerometer and gyroscope).
In order for us to solve the problem of the scaled camera pose and the problem of errors
accumulation of IMU’s integration when using any of these devices separately. There exist
many algorithms to perform sensor fusion between multi-sensor systems to keep the benefits
You sent January 10 at 12:00 AM
of all the fused sensors by updating the estimations of the higher frequency sensor (IMU in our
case) using the readings from the lower frequency sensor (Camera in our case).
The main advantage of these fusion frameworks is that the estimations are calculated with
respect to the inertial frames of both the sensors; World (w) frame for the IMU and Vision (v)
frame for the camera instead of their body frames (i), (c) respectively. Using the inertial frames
as a reference for our estimations enables us to obtain consistent estimations with no visual
drifts or scale factor.
Implementation Tasks
1. Select a dataset from (TUM, n.d.) containing camera frames and IMU readings.
2. Implement the Structure from motion Algorithm (Sfm) (Motion, n.d.) on the camera frames
pipeline to obtain the scaled pose and plot the position (x,y,z) and euler angles (ø, θ,Ψ) for all
the time steps with respect to the Camera body frame (c).
Hint: For the Sfm Algorithm, you can use the MATLAB algorithm (MATLAB, n.d.)but you need
to configure it to work efficiently with your dataset.
3. Implement a powerful (accurate and fast) quaternion integration (Solà, 2015) algorithm for
the IMU (gyroscope) readings only to obtain the euler angles (ø, θ,Ψ) and plot them for all the
time steps with respect to the IMU body frame (i).
4. Investigate (just research (Weiss, 2013)) the ways (filters) used to fuse the IMU/Camera
readings in a Visual-Inertial Odometry framework.

## Visual SLAM

<p align="center">
  <img width="300" height="300" src="https://user-images.githubusercontent.com/47057759/105985611-f74abd80-609b-11eb-8ebb-63ddc0f83bba.png">
</p>
## Extended KALMAN FILTER
<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/47057759/105985300-94592680-609b-11eb-8f37-dfc648e11d0b.png">
</p>
The extended Kalman filter algorithm is similar to the Kalman filter algorithm except the linear functions in Kalman filter are replaced by their non-linear generalizations in EKF.

## Structure from Motion
Structure from motion is a special technique in computer vision for reconstruction
three-dimensional models from a sequence of two-dimensional images by analyzing the
motion between these images. This method allows computers to estimate distances to
objects using camera images

<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/47057759/105987535-b1dbbf80-609e-11eb-94a1-043289d6e0d6.png">
</p>
Given: m images of n fixed 3D points
𝑋_𝑖𝑗=𝑃_𝑖 𝑋_𝑗       𝑖=1,..,𝑚   𝑗=1,…,𝑛
Problem: estimate 𝑚  projection matrices 𝑃_𝑖 and 𝑛 3D points 𝑋_𝑗 from the 𝑚𝑛 correspondences 𝑋_𝑖𝑗 <br>
Fundamental matrix maps from a point in one image to a line in the other 𝐼^′=𝐹𝑥 and 𝐼=𝐹^𝑇 𝑥′
If x and x’ correspond to the same 3d point X:  𝑥^′𝑇 𝐹𝑥=0
### Incremental SfM

<p align="center">
  <img width="500" height="100" src="https://user-images.githubusercontent.com/47057759/105987774-0b43ee80-609f-11eb-94a6-1c4487ed53d4.png">
</p>
## Extended Kalman Filter Algorithm

<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/47057759/105988072-6ece1c00-609f-11eb-901d-7f568ed5222c.png">
</p>
## Results

## Conclusion & Future Works

## References
