from robot import Robot
import numpy as np
import argparse
import cv2
import os

def main(args):
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    mode = args.mode
    config_file = args.config_file
    robot = Robot(workspace_limits, args.config_file, fixed_obj_coord=True)

    ## visualization mode 
    if mode == 0:
        cnt = 0
        while True:
            rgb_img, depth_img = robot.get_camera_data()
            if cnt >= 10:
                cv2.imshow('rgb_img', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)
                if key != -1:
                    break
            cnt += 1

    ## dataset mode
    elif mode == 1:
        cnt = 0
        img_num = 0
        path = "./vision/object_detection/yolo/labelImg/data/train"

        if not os.path.isdir(path):
            os.makedirs(path)

        for i in range(15):
            rgb_img, _ = robot.get_camera_data()
            if cnt >= 10:
                img_num += 1
                cv2.imwrite(path + "/img_{:06}.jpg".format(img_num), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cnt += 1

    ## grasping mode
    elif mode == 2:
        # start grasping
        position = np.array([-0.6018, -0.1002, 0.05])
        orientation = 30
        robot.grasp(position, orientation, workspace_limits)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Object options ---------------
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--config_file', dest='config_file', action='store', default='simulation/random/random-1blocks.txt')
    
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)