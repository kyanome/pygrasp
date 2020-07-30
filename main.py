from robot import Robot
import numpy as np
import argparse

def main(args):
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    config_file = args.config_file
    robot = Robot(workspace_limits, args.config_file, fixed_obj_coord=True)

    # start grasping
    position = np.array([-0.6273, -0.127, 0.05])
    orientation = 30
    robot.grasp(position, orientation, workspace_limits)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Object options ---------------
    parser.add_argument('--config_file', dest='config_file', action='store', default='simulation/random/random-1blocks.txt')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)