from modules import get_prediction, object_detection_tracking, read_write_video
from centroid_tracker import CentroidTracker

import argparse, random
from torchvision import models
import cv2, torch, numpy as np


#########################################
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
#########################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Считаем входной видео-поток и объявим выходной путь
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
args = vars(ap.parse_args())

image = cv2.imread(args['input'])

###################-- FasterRCNN--#######################
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

#############-Centroid-Tracker-###########################
ct = CentroidTracker()

# output = object_detection_tracking(image, ct, model)
# cv2.imwrite(args['output'], output)
read_write_video(args['input'], args['output'], model, ct)
