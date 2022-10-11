import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

"""
SINGLE IMAGE TESTING:
This script creates 30 black images and sends them to the net to clean the buffer. 
Then a single image IMAGE_PATH is sent repeatedly. 
If SHIFT, the image sent is shifted by some pixel to simulate the movement.  
Check the output with rviz.
"""

IMAGE_PATH = "/workspace/test_images/cam_img_00038_00001438.png"
SHIFT = False

rospy.init_node("tester")
cam_topic="/camera/camera/image"
pub = rospy.Publisher(cam_topic, Image, queue_size=10)
cv_bridge = CvBridge()
count = 0

fake_imgs = [np.ones((320, 320, 3)).astype(np.uint8) for i in range(30)]

for img in fake_imgs:
    ros_img = cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")
    pub.publish(ros_img)
    time.sleep(4)
    count += 1
    print(f"Fake n:{count}")



for f in range(35):
    img_to_send = cv2.imread(IMAGE_PATH)
    if SHIFT:
        shift = f % 20
        if shift > 0:
            if shift > 10:
                shift = 20 - shift
                img_to_send[:-shift, :, :] = img_to_send[shift:, :, :]
            else:
                img_to_send[:-shift, :, :] = img_to_send[shift:, :, :]
    ros_img = cv_bridge.cv2_to_imgmsg(img_to_send, encoding="passthrough")
    pub.publish(ros_img)
    time.sleep(4)
    count += 1
    print(f"Real Image n:{count}")