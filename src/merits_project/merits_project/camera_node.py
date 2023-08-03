#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from project_interfaces.msg import Multiimage
from cv_bridge import CvBridge

import numpy as np
import cv2 as cv
import time

class CamPublisher(Node):
    def __init__(self):
        super().__init__("camera_node")

        #initializes publisher
        self.publisher_depth = self.create_publisher(Multiimage, "stereo_cams", 10)
        self.get_logger().info("camera nodes have been started")

        #change these based on your camera indices
        self.declare_parameter("Left_cam_index", 2)
        self.declare_parameter("Right_cam_index", 0)
        
        #NOTE uncomment this to count frames published.
        #self.i = 0
        #initializes bridge to convert cv2 images to image messages

        self.bridge = CvBridge()
        self.cam_start()
    
    def cam_start(self):

        camL_index = self.get_parameter("Left_cam_index").value
        camR_index = self.get_parameter("Right_cam_index").value
        
        #captures videos
        capL = cv.VideoCapture(camL_index)
        capR = cv.VideoCapture(camR_index)

        while True:
            #reads cameras each frame and quits if either one fails.
            retL, frameL = capL.read()
            retR, frameR = capR.read()

            if not retL:
                print("Video error with left camera")
                break

            if not retR:
                print("Video error with right camera")
                break
            
            #converts cv2 images to img messages
            imgmsgL = self.bridge.cv2_to_imgmsg(frameL)
            imgmsgR = self.bridge.cv2_to_imgmsg(frameR)

            #initializes custom msg object and publishes both frames.
            both_cams = Multiimage()
            both_cams.cam_left = imgmsgL
            both_cams.cam_right = imgmsgR
            self.publisher_depth.publish(both_cams)

            #NOTE uncomment this to set a custom publishing rate in FPS.
            #time.sleep(.016666666) #publishes at 60 FPS

            #self.i += 1
            #print(str(self.i) + " frames published")

        capL.release()
        capR.release()
            
#basic ROS2 node spinning main function
def main(args=None):
    rclpy.init(args=args)
    node = CamPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
