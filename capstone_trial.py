"""
	capstone trial for video playing
"""

import jetson.inference
import sys
import cv2
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
# camera = jetson.utils.gstCamera(1280,720, "~/Downloads/Videos_for_Capstone/0326test_pet2")
# camera = jetson.utils.videoSource("0326test_pet.mp4", argv=sys.argv)
display = jetson.utils.videoOutput("my_video1.mp4", argv=sys.argv)

while display.IsStreaming():
    img, width, height = camera.CaptureRGBA()
    # img = camera.Capture()
    # detections = net.Detect(img)
    detections = net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
    # display.Render(img)
    # display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))


