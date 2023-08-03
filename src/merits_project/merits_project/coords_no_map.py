#!/usr/bin/env python3

#NOTE to see notes on how this all works, look at the coords_synchronized.py file.
#this is just that file without disparity map calculations.

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from project_interfaces.msg import Multiimage
from project_interfaces.msg import Xyz

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import numpy as np
import cv2 as cv

class CoordsCalculator(Node):
    def __init__(self):
        super().__init__("Coords_Calculator")

        self.depth_pub = self.create_publisher(Xyz, "obj_coords_cm", 10)
        self.subscriber = self.create_subscription(Multiimage, "stereo_cams", self.cam_callback, 10)

        self.declare_parameter("custom_x_axis", 0)
        self.declare_parameter("custom_y_axis", 0)
        self.declare_parameter("custom_z_axis", 0)
        self.declare_parameter("Remapping_file", "src/merits_project/merits_project/opencv_assets/stereoMapFML3.xml")
        self.declare_parameter("YOLO_file", "src/merits_project/merits_project/opencv_assets/best.pt")

        self.get_logger().info("subscriber nodes have been started")
        self.bridge = CvBridge()
        #self.counter = 0

    def find_depth(self, detect_left_x,  detect_right_x, left_y, frameL, frameR, Baseline, cam_angle, f, cam_cent_x, cam_cent_y):
    
        width_R = frameR.shape[1]
        width_L = frameL.shape[1]

        if width_R == width_L:
            pass
        else:
            self.get_logger().error("frames don't match")
            return

        x_param = self.get_parameter("custom_x_axis").value
        y_param = self.get_parameter("custom_y_axis").value
        z_param = self.get_parameter("custom_z_axis").value

        disp = abs(detect_right_x - detect_left_x)
        zdepth = (f * Baseline)/disp

        y_add = (np.tan(cam_angle * np.pi/180)) * zdepth + 5 #5 is in cm, distance from bottom of wooden block to camera lens
        error_y = 0.0131 * zdepth + 0.132
        error_x = 0.0705 * zdepth - 2.26

        self.get_logger().info(str(left_y))

        x = abs(((zdepth * (detect_left_x - cam_cent_x) / f) + error_x) - 22.26) + x_param #changes x-axis 0 location
        y = abs((zdepth * (left_y - cam_cent_y) / f) - y_add) - error_y - y_param #changes y-axis 0 location to bottom of blocks

        zdepth = zdepth - z_param

        return (abs(zdepth), abs(x), abs(y))
    
    def cam_callback(self, msg):
        
        camL = self.bridge.imgmsg_to_cv2(msg.cam_left)
        camR = self.bridge.imgmsg_to_cv2(msg.cam_right)

        filtered_disp_vis = None

        center_l_x = 380.5
        center_l_y = 238.7
        cam_angle = 15 #angle the camera is tilted up
        Focal_Length_pixel = 627.9
        Baseline_cm = 9.5

        cv_file = cv.FileStorage(f"{self.get_parameter('Remapping_file').value}", cv.FILE_STORAGE_READ)
        Left_Stereo_Map_x = cv_file.getNode('stereomapL_x').mat()
        Left_Stereo_Map_y = cv_file.getNode('stereomapL_y').mat()
        Right_Stereo_Map_x = cv_file.getNode('stereomapR_x').mat()
        Right_Stereo_Map_y = cv_file.getNode('stereomapR_y').mat()

        cv_file.release()

        Left_final = cv.remap(camL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        Right_final = cv.remap(camR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

        model = YOLO(f"{self.get_parameter('YOLO_file').value}")

        imgR = cv.cvtColor(Right_final, cv.COLOR_BGR2RGB)
        imgL = cv.cvtColor(Left_final, cv.COLOR_BGR2RGB)

        resultsR = model.predict(imgR)
        resultsL = model.predict(imgL)
        results_unmapL = model.predict(camL)

        coordsL = None
        coordsR = None

        if (len(resultsR) or len(resultsL)) == 0:
            pass
        else:    
            r = resultsR[0]
            annotatorR = Annotator(Right_final)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                conf = round(box.conf.item(), 2)
                detect_right_x = round(((float(b[0].item()) + float(b[2].item()))/2), 2)
                annotatorR.box_label(b, model.names[int(c)] + f" (conf = {conf})")
                coordsR = True

            frame_r_show = annotatorR.result()
            #cv.imshow("right_detect", frame_r_show)
            cv.imshow("right_detect", cv.resize(frame_r_show, (0,0), fx=0.5, fy=0.5))

            
            r2 = resultsL[0]
            annotatorL = Annotator(Left_final)

            boxes2 = r2.boxes
            for box in boxes2:
                b = box.xyxy[0]
                c = box.cls
                conf = round(box.conf.item(), 2)
                detect_left_x = round(((float(b[0].item()) + float(b[2].item()))/2), 2)
                annotatorL.box_label(b, model.names[int(c)] + f" (conf = {conf})")
                coordsL = True

            frame_l_show = annotatorL.result()
            #cv.imshow("left_detect", frame_l_show)
            cv.imshow("left_detect", cv.resize(frame_l_show, (0,0), fx=0.5, fy=0.5))

            r3 = results_unmapL[0]
            boxes3 = r3.boxes
            for box in boxes3:
                b = box.xyxy[0]
                left_y = round(((float(b[1].item()) + float(b[3].item()))/2), 2)
            
        if coordsL is not None and coordsR is not None and filtered_disp_vis is not None:
            depth, x, y = self.find_depth(detect_left_x, detect_right_x, left_y, Left_final, Right_final, Baseline_cm, cam_angle, Focal_Length_pixel, center_l_x, center_l_y)
            
            coords_msg  = Xyz()
            coords_msg.x = x
            coords_msg.y = y
            coords_msg.z = depth

            self.depth_pub.publish(coords_msg)

            window = np.zeros((500, 500, 3), np.uint8)
            cv.putText(window, f"Object Head Coordinates", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(window, f"Depth (Z) of object: {round(depth, 2)}cm", (0, 350), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(window, f"X of object: {round(x, 2)} cm", (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(window, f"Y of object: {round(y, 2)} cm", (0, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            
            #cv.imshow("objcoords", window)
            cv.imshow("objcoords", cv.resize(window, (0,0), fx=0.5, fy=0.5))

        elif coordsL is None or coordsR is None or filtered_disp_vis is None:
            window = np.zeros((500, 500, 3), np.uint8)
            cv.putText(window, "WARNING: TRACKING LOST!", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            
            #cv.imshow("objcoords", window)
            cv.imshow("objcoords", cv.resize(window, (0,0), fx=0.5, fy=0.5))

        #cv.imwrite(f"src/merits_project/merits_project/depth_frame/depth{self.counter}.jpg", filtered_disp_vis)
        #cv.imwrite(f"src/merits_project/merits_project/camR/imgR{self.counter}.jpg", frame_r_show)
        #cv.imwrite(f"src/merits_project/merits_project/camL/imgL{self.counter}.jpg", frame_l_show)
        #cv.imwrite(f"src/merits_project/merits_project/window/coords{self.counter}.jpg", window)

        #self.counter += 1

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CoordsCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    
