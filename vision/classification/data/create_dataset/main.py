from darknet import YOLO
import argparse
from cv_utils import *
from utils import *

def main(args):
	cfg = args.cfg
	obj_data = args.obj_data
	weights = args.weights
	img_path = args.img_path

	# create save dir and anno file
	labels = {"pen", "box", "bottle", "spoon"}
	create_img_dir(labels)
	create_anno_file()

	# read an image.
	frame = cv2.imread(img_path)

	# setup yolo model
	yolo = YOLO()
	net = yolo.load_net(cfg.encode(), weights.encode(), 0)
	meta = yolo.load_meta(obj_data.encode())

	# inference
	r = yolo.detect(net, meta, img_path.encode())

	# post process
	for i, r0 in enumerate(r):
	    bbox_info = get_bbox(r0)
	    bbox_img, label  = get_bbox_img(frame, bbox_info)
	    save_img(bbox_img, label, i)
	    cv2.imshow("bbox img", bbox_img)
	    cv2.waitKey(0)

	cv2.destroyAllWindows()


if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()
	# --------------- Object options ---------------
	parser.add_argument('--img_path', dest='img_path', action='store', default='data/train/img_000001.jpg')
	parser.add_argument('--cfg', dest='cfg', action='store', default='cfg/yolov2.cfg')
	parser.add_argument('--obj_data', dest='obj_data', action='store', default='cfg/obj.data')
	parser.add_argument('--weights', dest='weights', action='store', default='backup/yolov2_10000.weights')

	args = parser.parse_args()
	main(args)


	