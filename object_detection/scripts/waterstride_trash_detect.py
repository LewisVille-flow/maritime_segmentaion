#!/usr/bin/env python
#-*- coding:utf-8 -*-

import rospy
import sys

from geometry_msgs.msg import Pose2D
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import BoundingBox2DArray

from std_msgs.msg import Int32
from object_detection.msg import Direction

import jetson.inference
import jetson.utils


def main():
    # ROS setting
    pub = rospy.Publisher('trash_direction', Direction, queue_size=30)
    rate = rospy.Rate(30)

    # Camera setting
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
    camera = jetson.utils.gstCamera(1280, 720, "/dev/video2")
    display = jetson.utils.glDisplay()

    # ros pub iteration
    while not rospy.is_shutdown() and display.IsOpen():
	# Camera setting again
        img, width, height = camera.CaptureRGBA()
        detections = net.Detect(img, width, height)
	
	# Direction setting
	left_area_end = width /3
	forward_area_end = left_area_end *2
	right_area_end = width

        for det in detections:
	    # Custom msgs
	    direction_msg = Direction()
	   
	    # Direction
	    if(det.Center[0] < left_area_end):
		direction_msg.direction = "LEFT"
	    elif(det.Center[0] < forward_area_end):
		direction_msg.direction = "FORWARD"
	    else:
		direction_msg.direction = "RIGHT"

	    # Direction_helper(percent)
	    if(det.Center[0] < (width/2)):
	    	direction_msg.percent = -(1 -(det.Center[0] / (width/2)))
	    else:
	    	direction_msg.percent = (det.Center[0] / (width/2)) -1

	    # Publish
            pub.publish(direction_msg)

        display.RenderOnce(img, width, height)
        display.SetTitle("Trash Detection | Network {:.0f}FPS".format(net.GetNetworkFPS()))

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('object_detection_node', anonymous=True)

    try:
        main()
    except rospy.ROSInterruptException:
        sys.exit(0)
