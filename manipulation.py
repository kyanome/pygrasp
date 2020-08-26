from robot import Robot
import numpy as np
import argparse
import cv2
import os

def depth2pointcloud(K, depth_map, pixel_x, pixel_y):
    focal_length_x = K[0, 0]
    focal_length_y = K[1, 1]
    principal_point_x = K[0, 2]
    principal_point_y = K[1, 2]

    depth_value = depth_map[pixel_y, pixel_x]
    x = (pixel_x - principal_point_x)*depth_value / focal_length_x
    y = (pixel_y - principal_point_y)*depth_value / focal_length_y

    return np.array([x, y, depth_value])

def main(args):
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    mode = args.mode
    config_file = args.config_file
    robot = Robot(workspace_limits, args.config_file, fixed_obj_coord=True)

    cnt = 0
    while True:
        rgb_img, depth_img = robot.get_camera_data()
        if cnt >= 10:
            point_cloud = depth2pointcloud(robot.cam_intrinsics, depth_img, 445, 355)
            ur5_position = robot.get_ur5_position()
            target_point = point_cloud - ur5_position
            target_point[0] = -target_point[0]
            orientation = 30
            from matplotlib import pyplot as plt
            plt.imshow(depth_img, cmap="plasma")
            plt.show()
            robot.grasp(target_point, orientation, workspace_limits)
            import pdb; pdb.set_trace()
            cv2.imshow('rgb_img', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            if key != -1:
                break
        cnt += 1


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Object options ---------------
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--config_file', dest='config_file', action='store', default='simulation/random/random-1blocks.txt')
    
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)