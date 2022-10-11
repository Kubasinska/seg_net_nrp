import cv2
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

"""
REAL VIDEO SENDER:
This script sends all the frames in video.webm one by one. 
"""

rospy.init_node("tester")
cam_topic="/camera/camera/image"
pub = rospy.Publisher(cam_topic, Image, queue_size=10)
cv_bridge = CvBridge()
vidcap = cv2.VideoCapture('test.webm')
success,image = vidcap.read()
count = 0

while success:
    ros_img = cv_bridge.cv2_to_imgmsg(image, encoding="passthrough")
    pub.publish(ros_img)
    time.sleep(4)
    success,image = vidcap.read()
    print('Read a new frame: ', success, count)
    count += 1