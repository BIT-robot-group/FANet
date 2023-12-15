from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import sys
import time
# import rospy
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray

import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

# transformation from the robot base to aruco tag
M_BL = np.array([[1., 0., 0.,  0.30000],
                 [0., 1., 0.,  0.32000],
                 [0., 0., 1.,  -0.0450],
                 [0., 0., 0.,  1.00000]])

# default transformation from the camera to aruco tag
default_M_CL = np.array([[-0.07134498, -0.99639369,  0.0459293,  -0.13825178],
                         [-0.8045912,   0.03027403, -0.59305689,  0.08434352],
                         [ 0.58952768, -0.07926594, -0.8038495,   0.66103522],
                         [ 0.,          0.,          0.,          1.        ]]
                        )

# camera intrinsic matrix of Realsense D435
cameraMatrix = np.array([[607.47165, 0.0,  325.90064],
                         [0.0, 606.30420, 240.91934],
                         [0.0, 0.0, 1.0]])

# distortion of Realsense D435
distCoeffs = np.array([0.08847, -0.04283, 0.00134, -0.00102, 0.0])




def pre_process(rgb_img, depth_img):
    inp_image = rgb_img.copy()
    inp_image[:, :, 0] = depth_img
    inp_image = cv2.resize(inp_image, (512, 512))
    inp_image = inp_image[:, :, ::-1]

    return inp_image

def isWithinRange(pxl, w, h):
    x, y = pxl[:]

    return w/12. <= x <= 11*w/12 and h/12. <= y <= 11*h/12

def KpsToGrasppose(net_output, rgb_img, depth_map,  M_BL, cameraMatrix, visualize=True):
    kps_pr = []
    for category_id, preds in net_output.items():
        if len(preds) == 0:
            continue

        for pred in preds:
            kps = pred[:4]
            score = pred[-1]
            kps_pr.append([kps[0], kps[1], kps[2], kps[3], score])

    # no detection
    if len(kps_pr) == 0:
        return [0, 0, 0, 0]

    # sort by the confidence score
    kps_pr = sorted(kps_pr, key=lambda x: x[-1], reverse=True)
    # select the top 1 grasp prediction within the workspace
    res = None
    for kp_pr in kps_pr:
        f_w, f_h = 1024. / 512., 1024. / 512.
        kp_lm = (int(kp_pr[0] * f_w), int(kp_pr[1] * f_h))
        kp_rm = (int(kp_pr[2] * f_w), int(kp_pr[3] * f_h))

        if isWithinRange(kp_lm, 1024, 1024) and isWithinRange(kp_rm, 1024, 1024):
            res = kp_pr
            break

    if res is None:
        return [0, 0, 0, 0]

    f_w, f_h = 1024./512., 1024./512.
    kp_lm = (int(res[0]*f_w), int(res[1]*f_h))
    kp_rm = (int(res[2]*f_w), int(res[3]*f_h))
    center = (int((kp_lm[0]+kp_rm[0])/2), int((kp_lm[1]+kp_rm[1])/2))

    # rgb_img = cv2.resize(rgb_img,(512,512))
    # draw arrow for left-middle and right-middle key-points
    lm_ep = (int(kp_lm[0] + (kp_rm[0] - kp_lm[0]) / 5.), int(kp_lm[1] + (kp_rm[1] - kp_lm[1]) / 5.))
    rm_ep = (int(kp_rm[0] + (kp_lm[0] - kp_rm[0]) / 5.), int(kp_rm[1] + (kp_lm[1] - kp_rm[1]) / 5.))
    rgb_img = cv2.arrowedLine(rgb_img, kp_lm, lm_ep, (0, 0, 0), 2)
    rgb_img = cv2.arrowedLine(rgb_img, kp_rm, rm_ep, (0, 0, 0), 2)
    # draw left-middle, right-middle and center key-points
    rgb_img = cv2.circle(rgb_img, (int(kp_lm[0]), int(kp_lm[1])), 2, (0, 255, 0), 2)
    rgb_img = cv2.circle(rgb_img, (int(kp_rm[0]), int(kp_rm[1])), 2, (0, 255, 0), 2)
    rgb_img = cv2.circle(rgb_img, (int(center[0]), int(center[1])), 2, (0, 0, 255), 2)

    if visualize:
        cv2.namedWindow('visual', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('visual', rgb_img)

    # return [center_3d[0], center_3d[1], center_3d[2], orientation, dist]

def run(opt,  img, depth):

    start_time = time.time()
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    # print(opt)
    Detector = detector_factory[opt.task]

    detector = Detector(opt)

    for i in range(4):
        
        inp_image = pre_process(img, depth)


        # pass the image into the network
        ret = detector.run(inp_image)

        ret = ret["results"]

        # img = cv2.resize(img,(512,512))
        KpsToGrasppose(ret, img, depth,  M_BL, cameraMatrix)

        # msg = Float64MultiArray()
        # msg.data = loc_ori
        # pub_res.publish(msg)

        k = cv2.waitKey(0)
        if k == ord('q'):
            break

    # video_saver.release()
    cv2.destroyAllWindows()
    # Stop streaming
    # pipeline.stop()

if __name__ == '__main__':
    opt = opts().parse()

    img = cv2.imread('/home/yusheng/jacquard/ac00e79c78c266eba3b36c8802b3ab55/0_ac00e79c78c266eba3b36c8802b3ab55_RGB.png')
    depth = cv2.imread('/home/yusheng/jacquard/ac00e79c78c266eba3b36c8802b3ab55/0_ac00e79c78c266eba3b36c8802b3ab55_perfect_depth.tiff',2)
    
    # img = cv2.resize(img,(512,512))
    # depth = cv2.resize(depth,(512,512))
    # depth = np.expand_dims(depth, axis = 2)
    # print(img.shape)

    # Configure depth and color streams
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # # Start streaming
    # profile = pipeline.start(config)

    # # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: ", depth_scale)

    # align_to = rs.stream.color
    # align = rs.align(align_to)

    # initialize ros node
    # rospy.init_node("Static_grasping")
    # # Publisher of perception result
    # pub_res = rospy.Publisher('/result', Float64MultiArray, queue_size=10)

    run(opt, img, depth)