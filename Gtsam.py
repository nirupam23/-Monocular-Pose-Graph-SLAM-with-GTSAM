import cv2
import numpy as np
import gtsam
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

###############################################################################
#                           UTILITY FUNCTIONS
###############################################################################

def video_to_frames(video_path, max_frames=None):
    """
    Reads a video from the given path, returns a list of frames (BGR format).
    :param video_path: Full path to your drone video file.
    :param max_frames: If set, limit the total frames read.
    :return: List of frames as NumPy arrays (BGR).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return frames

def cv_pose_to_gtsam(R, t):
    """
    Convert a rotation (3x3) and translation (3x1 or 3,) from OpenCV
    to a GTSAM Pose3. Ensures t is flattened to shape (3,).
    """
    t = t.ravel()  # Flatten to (3,)
    rot = gtsam.Rot3(R)
    trans = gtsam.Point3(t[0], t[1], t[2])
    return gtsam.Pose3(rot, trans)

def visualize_matches(frame1, kp1, frame2, kp2, matches, loop_key, current_frame):
    """
    Visualize matches between two frames.
    :param frame1: First image (BGR).
    :param kp1: Keypoints from the first image.
    :param frame2: Second image (BGR).
    :param kp2: Keypoints from the second image.
    :param matches: List of matched cv2.DMatch objects.
    :param loop_key: Index of the keyframe.
    :param current_frame: Index of the current frame.
    """
    # Draw matches
    matched_image = cv2.drawMatches(frame1, kp1, frame2, kp2, matches, None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Convert BGR to RGB for matplotlib
    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    
    # Display the matched image
    plt.figure(figsize=(20, 10))
    plt.title(f'Loop Closure: Frame {current_frame} ↔ Keyframe {loop_key}')
    plt.imshow(matched_image)
    plt.axis('off')
    plt.show()

###############################################################################
#                           MAIN PIPELINE WITH LOOP CLOSURE
###############################################################################

def main():
    #-----------------------------------------------------------------------
    # 1. SET VIDEO PATH & CAMERA INTRINSICS
    #-----------------------------------------------------------------------
    video_path = r"C:\Users\Nirup\Downloads\drone2.mp4"  # <--- Your drone video path
    max_frames = 600  # Limit frames for the demo; adjust as desired

    # Read frames from the drone video
    frames = video_to_frames(video_path, max_frames)
    num_frames = len(frames)
    print(f"[INFO] Total frames loaded: {num_frames}")
    if num_frames < 2:
        print("[ERROR] Not enough frames to process.")
        return

    # Camera intrinsics (PLACEHOLDER values!)
    fx = 700.0
    fy = 700.0
    cx = 640.0 / 2.0
    cy = 360.0 / 2.0
    camera_matrix = np.array([
        [fx,  0,   cx],
        [0,   fy,  cy],
        [0,   0,   1]
    ], dtype=np.float64)

    #-----------------------------------------------------------------------
    # 2. FEATURE DETECTOR & MATCHER (ORB + BFMatcher)
    #-----------------------------------------------------------------------
    orb = cv2.ORB_create(nfeatures=2000)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #-----------------------------------------------------------------------
    # 3. GTSAM FACTOR GRAPH + INITIAL ESTIMATE
    #-----------------------------------------------------------------------
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Noise models
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))  # Tighter noise for loop closures

    # Add a prior on the first pose (key=0) to fix gauge freedom
    first_pose = gtsam.Pose3()  # Identity
    graph.add(gtsam.PriorFactorPose3(0, first_pose, prior_noise))
    initial_estimate.insert(0, first_pose)

    current_pose_guess = first_pose  # We'll accumulate relative transforms

    #-----------------------------------------------------------------------
    # 4. PROCESS FRAMES PAIRWISE & ADD FACTORS WITH LOOP CLOSURE
    #-----------------------------------------------------------------------
    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    # Keyframe settings
    keyframe_interval = 10  # Add a keyframe every 10 frames
    keyframes = {}  # Store keyframe descriptors and indices
    keyframes[0] = (prev_kp, prev_des)

    for i in range(1, num_frames):
        current_frame = frames[i]
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 4.1 Detect & compute features
        kp, des = orb.detectAndCompute(gray_current, None)

        # 4.2 Match features with previous frame
        if prev_des is None or des is None:
            print(f"[WARN] Missing descriptors at frame {i}, skipping factor.")
            prev_kp, prev_des = kp, des
            continue

        matches = bf_matcher.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:200]  # pick top 200 for speed

        print(f"Frame {i-1} -> {i}: Found {len(matches)} matches, using {len(good_matches)} best.")

        # 4.3 Prepare point sets for essential matrix
        if len(good_matches) < 8:
            print(f"[WARN] Not enough good matches at frame {i}, skipping factor.")
            prev_kp, prev_des = kp, des
            continue

        pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches])

        # 4.4 Estimate the Essential Matrix & relative pose
        E, inliers = cv2.findEssentialMat(
            pts_prev, pts_curr, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            print(f"[WARN] Degenerate E matrix at frame {i}, skipping factor.")
            prev_kp, prev_des = kp, des
            continue

        # recoverPose -> rotation (3x3), translation (3x1) up to scale
        _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, camera_matrix)

        # Convert (R, t) to GTSAM Pose3
        relative_pose = cv_pose_to_gtsam(R, t)

        # 4.5 Add a BetweenFactorPose3: from (i-1) to i
        from_symbol = i - 1
        to_symbol   = i
        graph.add(gtsam.BetweenFactorPose3(from_symbol, to_symbol, relative_pose, odometry_noise))

        # 4.6 Update current guess of absolute pose
        current_pose_guess = current_pose_guess.compose(relative_pose)
        initial_estimate.insert(to_symbol, current_pose_guess)

        #-------------------------------------------------------------------
        # 4.7 LOOP CLOSURE DETECTION
        #-------------------------------------------------------------------
        loop_found = False
        loop_key = -1
        if i % keyframe_interval == 0:
            # Add current frame as a keyframe
            keyframes[i] = (kp, des)

            # Attempt to find loop closure with previous keyframes
            for kf_idx, (kf_kp, kf_des) in keyframes.items():
                if abs(kf_idx - i) < keyframe_interval:
                    # Skip nearby keyframes to avoid false positives
                    continue

                # Match current descriptors with keyframe descriptors
                loop_matches = bf_matcher.match(des, kf_des)
                loop_matches = sorted(loop_matches, key=lambda x: x.distance)
                loop_good_matches = [m for m in loop_matches if m.distance < 30]  # Threshold can be tuned

                if len(loop_good_matches) > 100:  # Threshold for loop closure
                    loop_found = True
                    loop_key = kf_idx
                    print(f"[INFO] Loop closure detected between frame {i} and keyframe {kf_idx} with {len(loop_good_matches)} matches.")
                    break

            if loop_found:
                # Extract matched points
                pts_current = np.float32([kp[m.queryIdx].pt for m in loop_good_matches])
                pts_keyframe = np.float32([keyframes[loop_key][0][m.trainIdx].pt for m in loop_good_matches])

                # Estimate Essential Matrix for loop closure
                E_loop, inliers_loop = cv2.findEssentialMat(
                    pts_current, pts_keyframe, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
                )

                if E_loop is not None and E_loop.shape == (3, 3):
                    _, R_loop, t_loop, _ = cv2.recoverPose(E_loop, pts_current, pts_keyframe, camera_matrix)

                    # Convert to GTSAM Pose3
                    loop_relative_pose = cv_pose_to_gtsam(R_loop, t_loop)

                    # Add BetweenFactorPose3 for loop closure
                    graph.add(gtsam.BetweenFactorPose3(loop_key, to_symbol, loop_relative_pose, loop_noise))
                    print(f"[INFO] Added loop closure factor between frame {loop_key} and frame {i}.")

                    # Visualize the loop closure matches
                    frame_key = loop_key
                    frame_current = i
                    img_key = frames[frame_key]
                    img_current = frames[frame_current]

                    kp_key, des_key = keyframes[frame_key]
                    kp_current, des_current = kp, des

                    # Select the top matches for visualization
                    top_matches = sorted(loop_good_matches, key=lambda x: x.distance)[:50]  # Adjust number as needed

                    visualize_matches(img_key, kp_key, img_current, kp_current, top_matches, frame_key, frame_current)

                else:
                    print(f"[WARN] Failed to compute Essential Matrix for loop closure between frame {i} and keyframe {loop_key}.")

        # Prepare for next iteration
        prev_kp, prev_des = kp, des

    #-----------------------------------------------------------------------
    # 5. RUN OPTIMIZATION
    #-----------------------------------------------------------------------
    print("[INFO] Optimizing pose graph with Levenberg-Marquardt...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    print("[INFO] Optimization complete.")

    #-----------------------------------------------------------------------
    # 6. EXTRACT & VISUALIZE RESULTS
    #-----------------------------------------------------------------------
    poses_3d = []
    for key in range(num_frames):
        # Check if the result has a pose for this key
        if not result.exists(key):
            poses_3d.append([math.nan, math.nan, math.nan])
            continue

        try:
            # Attempt to retrieve pose as Pose3
            pose_i = result.atPose3(key)

            # Depending on the GTSAM build, pose_i might be a Pose3 or a numpy array
            if isinstance(pose_i, gtsam.Pose3):
                t_i = pose_i.translation()
                # Check if t_i has x(), y(), z() attributes
                if hasattr(t_i, 'x') and hasattr(t_i, 'y') and hasattr(t_i, 'z'):
                    poses_3d.append([t_i.x(), t_i.y(), t_i.z()])
                else:
                    # Assume it's a numpy array
                    poses_3d.append([t_i[0], t_i[1], t_i[2]])
            elif isinstance(pose_i, np.ndarray) and pose_i.shape == (4, 4):
                # Convert from 4x4 matrix to Pose3
                pose3 = gtsam.Pose3(pose_i)
                t_i = pose3.translation()
                poses_3d.append([t_i.x(), t_i.y(), t_i.z()])
            else:
                print(f"[WARN] Unexpected type at key {key}: {type(pose_i)}")
                poses_3d.append([math.nan, math.nan, math.nan])
        except AttributeError as e:
            print(f"[ERROR] AttributeError at key {key}: {e}")
            poses_3d.append([math.nan, math.nan, math.nan])
        except Exception as e:
            print(f"[ERROR] Unexpected error at key {key}: {e}")
            poses_3d.append([math.nan, math.nan, math.nan])

    poses_3d = np.array(poses_3d)

    # Plot the 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poses_3d[:,0], poses_3d[:,1], poses_3d[:,2], 'b-o', label='Optimized Poses')
    ax.set_title("Monocular Pose-Graph Trajectory with Loop Closure (Up to Scale)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    print("[INFO] Done. The above trajectory is up to scale and includes loop closures.")

#--------------------------------------------------------------------
# Execute main()
#--------------------------------------------------------------------
if __name__ == "__main__":
    main()
