#!/usr/bin/env python
#-*- coding:utf-8 -*-

import rospy
import sys

from geometry_msgs.msg import Pose2D
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import BoundingBox2DArray

from std_msgs.msg import Int32

import jetson.inference
import jetson.utils

def main():
    pub = rospy.Publisher('detect_results', BoundingBox2DArray, queue_size=10)
    
    pub_1 = rospy.Publisher('chatter', Int32, queue_size=10)
    rate= rospy.Rate(30)

    pub_msg = Int32()

    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
    camera = jetson.utils.gstCamera(1080, 720, "/dev/video2")
    display = jetson.utils.glDisplay()

    while not rospy.is_shutdown() and display.IsOpen():
        img, width, height = camera.CaptureRGBA()
        detections = net.Detect(img, width, height)

        #_list_class_id = []
        bbox_array = BoundingBox2DArray()
        for det in detections:
            bbox = BoundingBox2D()
            
            pub_msg.data = det.ClassID

            bbox.center = Pose2D()
            bbox.center.x = det.Center[0]
            bbox.center.y = det.Center[1]

            bbox.size_x = det.Width
            bbox.size_y = det.Height
            
            #_list_class_id.append(class_id)
            bbox_array.boxes.append(bbox)
            
            pub_1.publish(pub_msg)
            rospy.loginfo(rospy.get_caller_id() + 'Detected Class ID: %d' % pub_msg.data)


        #pub.publish(_list_class_id)
        pub.publish(bbox_array)
        
        display.RenderOnce(img, width, height)
        display.SetTitle("object Detection | Network {:.0f}FPS".format(net.GetNetworkFPS()))

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('object_detection_node', anonymous=True)

    try:
        main()
    except rospy.ROSInterruptException:
        sys.exit(0)
