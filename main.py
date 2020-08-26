from robot import Robot
import numpy as np
import argparse
import cv2
import os
import sys
import torch
sys.path.append('vision/classification/data/create_dataset')
from darknet_img import YOLO
from cv_utils import *
from utils import *
sys.path.append('vision/classification')
from network import Net
from trainer_utils import label_to_text


def main(args):
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    mode = args.mode
    config_file = args.config_file
    cfg = args.cfg
    obj_data = args.obj_data
    weights = args.weights

    robot = Robot(workspace_limits, args.config_file, fixed_obj_coord=True)

    # Load model
    ## yolo
    yolo = YOLO()
    yolo_net = yolo.load_net(cfg.encode(), weights.encode(), 0)
    meta = yolo.load_meta(obj_data.encode())
    ## cnn
    cnn_net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_net.to(device)
    save_path = "vision/classification/model/model.pt"
    cnn_net.load_state_dict(torch.load(save_path))

    cnt = 0
    for i in range(12):
        rgb_img, depth_img = robot.get_camera_data()
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        vis_img = rgb_img.copy()
        if cnt >= 10:
            r = yolo.detect(yolo_net, meta, rgb_img)
            for i in r:
                x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
                xmin, ymin, xmax, ymax = yolo.convertBack(float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(vis_img, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(vis_img, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 4)
            cv2.imshow("img", vis_img)

            # Classification
            for i, r0 in enumerate(r):
                bbox_info = get_bbox(r0)
                bbox_img, label  = get_bbox_img(rgb_img, bbox_info)
                torch_bbox_img = torch.from_numpy(bbox_img).float().to(device).permute(2,0,1).unsqueeze(0)
                output = cnn_net(torch_bbox_img)
                _, predicted = torch.max(output, 1)
                # show result
                print("category label = {}".format(label_to_text(predicted)))
                cv2.imshow("bbox img", bbox_img)
                cv2.waitKey(0)
        cnt += 1

     
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Object options ---------------
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--config_file', dest='config_file', action='store', default='simulation/random/random-1blocks.txt')
    parser.add_argument('--cfg', dest='cfg', action='store', default='vision/classification/data/create_dataset/cfg/yolov2.cfg')
    parser.add_argument('--obj_data', dest='obj_data', action='store', default='cfg/obj.data')
    parser.add_argument('--weights', dest='weights', action='store', default='vision/classification/data/create_dataset/backup/yolov2.backup')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)