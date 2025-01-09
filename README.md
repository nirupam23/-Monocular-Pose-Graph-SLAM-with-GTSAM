#Monocular Pose-Graph SLAM with GTSAM: Estimated an UAV’s 3D Flight Trajectory for PGO
This project implements a vision-based SLAM (Simultaneous Localization and Mapping) pipeline for UAVs using ORB features for keypoint detection and FLANN Matcher for feature matching across consecutive frames. The relative pose estimation is achieved by computing the Essential Matrix, and loop closures are detected by matching features with older keyframes. Loop closure constraints are incorporated into a factor graph using GTSAM, and all poses are optimized with the Levenberg-Marquardt algorithm. The resulting output includes a 3D flight trajectory visualization for the UAV along with visual matches that highlight the detected loop closures.
