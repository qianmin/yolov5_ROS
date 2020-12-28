#! /usr/bin/env python3


import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
IMAGE_WIDTH=1241
IMAGE_HEIGHT=376

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import cv2
import numpy as np
import time


IMAGE_WIDTH=1241
IMAGE_HEIGHT=376

rospy.init_node("listener", anonymous=True)
image_pubulish=rospy.Publisher('/camera/image_raw',Image,queue_size=1)
def publish_image(imgdata):
    image_temp=Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='rgb8'
    image_temp.data=np.array(imgdata).tostring()
    #print(imgdata)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=1241*3
    image_pub.publish(image_temp)
if __name__ == '__main__':

    rospy.init_node('pub_cv2_camera')
    image_pub = rospy.Publisher('/pub_cv2_camera_topic', Image, queue_size=1)

    cap=cv2.VideoCapture(0)
    while 1:
        img=cap.read()
        publish_image(img)
        #time.sleep(1)
        cv2.imshow('123',img)
        key=cv2.waitKey(1000)
        if key==ord('q'):
            break
        print("pubulish")
    rospy.spin()