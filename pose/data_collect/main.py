import argparse
import sys
import numpy as np
import threading
sys.path.append('../../')
from robot import Robot
import time

def worker1(robot):
    pose_list = []
    for i in range(60):
        orientation = robot.get_ur5_orientation()
        translation = robot.get_ur5_position()
        if (orientation[0]==0.0 and orientation[1]==0.0 and orientation[2]==0.0) or (translation[0]==0.0 and translation[1]==0.0 and translation[2]==0.0):
            pass
        else:
            pose = np.concatenate([translation, orientation])
            pose_list.append(pose)
    
    np.save('box',np.array(pose_list))

def worker2(robot):
    position = np.array([-0.6018, -0.1002, 0.05])
    orientation = 30
    robot.grasp(position, orientation, np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]))

def main(args):
    config_file = args.config_file
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])

    robot = Robot(workspace_limits, args.config_file, fixed_obj_coord=True)
    #print("ur5 orientation = {}".format(robot.get_ur5_orientation()))

    target_object = "pen"    
    if target_object=="pen":
        position = np.array([-0.6018, -0.1002, 0.05])
        orientation = 30
        robot.grasp(position, orientation, workspace_limits)
        robot.rotate_object(0, 0, workspace_limits)
        robot.rotate_object(1, 90, workspace_limits)
        position = robot.get_ur5_position()
        position[1] = position[1] - 0.1
        robot.move_object(position, None)
        robot.move_object(position, None)

    elif target_object=="spoon":
        position = np.array([-0.34529, -0.10752, 0.05])
        orientation = 30
        robot.grasp(position, orientation, workspace_limits)
        robot.rotate_object(1, 90, workspace_limits)
        robot.rotate_object(2, 150, workspace_limits)
        position = robot.get_ur5_position()
        position[1] = position[1] - 0.1
        position[2] = position[2] - 0.1
        robot.move_object(position, None)
        robot.rotate_object(2, 90, workspace_limits)

    elif target_object=="bottle":
        position = np.array([-0.58738, 0.079469, 0.15])
        orientation = 30
        robot.grasp(position, orientation, workspace_limits)
        robot.rotate_object(0, -10, workspace_limits)
        position = robot.get_ur5_position()
        position[1] = position[1] - 0.1
        robot.move_object(position, None)

    elif target_object=="box":
        position = np.array([-0.34226, 0.12401, 0.05])
        orientation = 30
        robot.grasp(position, orientation, workspace_limits)
        t1 = threading.Thread(target=worker1, args=(robot, ))
        t1.start()
        robot.rotate_object(1, 90, workspace_limits)
        position = robot.get_ur5_position()
        position[1] = position[1] - 0.1
        robot.move_to(position, None)
        
    #for i in range(100):
    #    print("ur5 orientation = {}".format(robot.get_ur5_orientation()))

if __name__ == "__main__":
    test = np.load("./data/bottle.npy")

    import pdb; pdb.set_trace()
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Object options ---------------
    parser.add_argument('--config_file', dest='config_file', action='store', default='../../simulation/random/random-1blocks.txt')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)