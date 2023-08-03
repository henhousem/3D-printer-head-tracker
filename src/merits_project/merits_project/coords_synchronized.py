#!/usr/bin/env python3

#imports all the necessary dependencies
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

#custom ROS2 messages used for this project.
from project_interfaces.msg import Multiimage
from project_interfaces.msg import Xyz

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import numpy as np
import cv2 as cv

class CoordsCalculator(Node):
    def __init__(self):
        super().__init__("Coords_Calculator")

        #initializes subscriber to camera frames and publisher of object coordinates
        self.depth_pub = self.create_publisher(Xyz, "obj_coords_cm", 10)
        self.subscriber = self.create_subscription(Multiimage, "stereo_cams", self.cam_callback, 10)

        #initializes parameters to change the zeros of each axis, the stereo camera remapping file,
        #the YOLOv8 object detection algorithm, and what type of disparity map you want.
        self.declare_parameter("custom_x_axis", 0)
        self.declare_parameter("custom_y_axis", 0)
        self.declare_parameter("custom_z_axis", 0)
        self.declare_parameter("Remapping_file", "src/merits_project/merits_project/opencv_assets/stereoMapSample2.xml")
        self.declare_parameter("YOLO_file", "src/merits_project/merits_project/opencv_assets/sampleYoloAlgorithm.pt")
        self.declare_parameter("Stereo_SGBM", True)

        self.get_logger().info("subscriber nodes have been started")
        self.bridge = CvBridge()

        #NOTE Uncomment this counter to save each frame
        #self.counter = 0

    #triangulation function where depth(Z position), X position, and Y position are calculated
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

        return (abs(zdepth), abs(x), (abs(y)))
    
    #callback that triggers when a set of images are received.
    def cam_callback(self, msg):
        
        #converts images to usable format
        camL = self.bridge.imgmsg_to_cv2(msg.cam_left)
        camR = self.bridge.imgmsg_to_cv2(msg.cam_right)

        #if any of these are None after the program runs, coordinates will not be traingulated.
        filtered_disp_vis = None
        coordsL = None
        coordsR = None
        left_y = None

        #parameter that determines whether the disparity map uses Stereo_SGBM or Stereo_BM
        stereo_sgbm = self.get_parameter("Stereo_SGBM").value

        #defines parameters used for triangulation
        center_l_x = 380.5 #optical center of left camera x position
        center_l_y = 238.7 #optical center of left camera y position
        cam_angle = 15 #angle the camera is tilted at
        Focal_Length_pixel = 627.9 #focal length of left camera
        Baseline_cm = 9.5 #baseline between the two camera lenses

        #loads in the remapping files for application
        cv_file = cv.FileStorage(f"{self.get_parameter('Remapping_file').value}", cv.FILE_STORAGE_READ)
        Left_Stereo_Map_x = cv_file.getNode('stereomapL_x').mat()
        Left_Stereo_Map_y = cv_file.getNode('stereomapL_y').mat()
        Right_Stereo_Map_x = cv_file.getNode('stereomapR_x').mat()
        Right_Stereo_Map_y = cv_file.getNode('stereomapR_y').mat()

        cv_file.release()

        if stereo_sgbm:
            
            #creates the StereoSGBM object and remaps the left and right images 
            #(to rectify and undistort them)
            stereo = cv.StereoSGBM.create(numDisparities=16, blockSize=11)
            Left_final = cv.remap(camL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
            Right_final = cv.remap(camR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        
            #parameter tuning for the disparity map
            numDisparities = 13 * 16
            minDisparity = 2 
            stereo.setNumDisparities(numDisparities)
            stereo.setBlockSize(0 * 2 + 5)
            stereo.setPreFilterCap(5)
            stereo.setP1(1 * 100)
            stereo.setP2(1 * 1000)
            stereo.setUniquenessRatio(3)
            stereo.setSpeckleRange(20)
            stereo.setSpeckleWindowSize(3 * 2 + 5)
            stereo.setDisp12MaxDiff(25)
            stereo.setMinDisparity(minDisparity)

            #lines 126-141 apply post-proccesing filters to the disparity map and then show it.
            wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)

            right_matcher = cv.ximgproc.createRightMatcher(stereo)

            left_disp = stereo.compute(Left_final, Right_final)
            right_disp = right_matcher.compute(Right_final, Left_final)

            wls_filter.setLambda(8000)
            wls_filter.setSigmaColor(1.5)

            filtered_disp = wls_filter.filter(left_disp, camL, disparity_map_right=right_disp)

            vis_mult = 16
            filtered_disp_vis = (filtered_disp / vis_mult).astype(np.uint8)
            #cv.imshow("filtered disparity", filtered_disp_vis)
            cv.imshow("filtered disparity", cv.resize(filtered_disp_vis, (0,0), fx=0.5, fy=0.5))

        else:
            #same processes, but for StereoBM instead of StereoSGBM
            stereo = cv.StereoBM_create(numDisparities=16, blockSize=5)

            gray_L = cv.cvtColor(camL, cv.COLOR_BGR2GRAY)
            gray_R = cv.cvtColor(camR, cv.COLOR_BGR2GRAY)

            Left_final = cv.remap(gray_L, Left_Stereo_Map_x, Left_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
            Right_final = cv.remap(gray_R, Right_Stereo_Map_x, Right_Stereo_Map_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

            numDisparities = 9
            minDisparity = 4
                                   
            stereo.setNumDisparities(numDisparities * 16)
            stereo.setBlockSize(11 * 2 + 5)
            stereo.setPreFilterType(1)
            stereo.setPreFilterSize(5 * 2 + 5)
            stereo.setPreFilterCap(6)
            stereo.setTextureThreshold(20)
            stereo.setUniquenessRatio(5)
            stereo.setSpeckleRange(28)
            stereo.setSpeckleWindowSize(3 * 2 + 5)
            stereo.setDisp12MaxDiff(5)
            stereo.setMinDisparity(minDisparity)


            wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)

            right_matcher = cv.ximgproc.createRightMatcher(stereo)

            left_disp = stereo.compute(Left_final, Right_final)
            right_disp = right_matcher.compute(Right_final, Left_final)

            wls_filter.setLambda(8000)
            wls_filter.setSigmaColor(1.5)

            filtered_disp = wls_filter.filter(left_disp, camL, disparity_map_right=right_disp)
            vis_mult = 16

            filtered_disp_vis = (filtered_disp / vis_mult).astype(np.uint8)
            
            #cv.imshow("filtered disparity", filtered_disp_vis)
            cv.imshow("filtered disparity", cv.resize(filtered_disp_vis, (0,0), fx=0.5, fy=0.5))

        #loads YOLO model
        model = YOLO(f"{self.get_parameter('YOLO_file').value}")

        #converts images to RGB for prediction
        imgR = cv.cvtColor(Right_final, cv.COLOR_BGR2RGB)
        imgL = cv.cvtColor(Left_final, cv.COLOR_BGR2RGB)

        #model prediction on each image
        resultsR = model.predict(imgR)
        resultsL = model.predict(imgL)
        results_unmapL = model.predict(camL)

        #if it gets no results, then skip the annotation
        if (len(resultsR) or len(resultsL)) == 0:
            pass
        else:   
            #this takes the first bounding box as a result, draws the box on
            #the right remapped frame with confidence, and stores the center x coordinate.
            #(also shows the image)
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

            
            #does the exact same for the left remapped frame
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

            #due to curves introduced by remapping files,
            #this makes the y-value of the center of the 3D printer head stay
            #constant with left or right movements.

            r3 = results_unmapL[0]
            boxes3 = r3.boxes
            for box in boxes3:
                b = box.xyxy[0]
                left_y = round(((float(b[1].item()) + float(b[3].item()))/2), 2)
                
        
        #again, does not use triangulation if any of the coords/disparity map 
        #fails to be reassigned.
        if coordsL is not None and coordsR is not None and left_y is not None and filtered_disp_vis is not None:
            depth, x, y = self.find_depth(detect_left_x, detect_right_x, left_y, Left_final, Right_final, Baseline_cm, cam_angle, Focal_Length_pixel, center_l_x, center_l_y)
            
            #uses custom msg type to publish coordinates.
            coords_msg = Xyz()
            coords_msg.x = x
            coords_msg.y = y
            coords_msg.z = depth
            self.depth_pub.publish(coords_msg)

            #draws a window with 3 coordinate values shown, then displays it.
            window = np.zeros((500, 500, 3), np.uint8)
            cv.putText(window, f"Object Head Coordinates", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(window, f"Depth (Z) of object: {round(depth, 2)}cm", (0, 350), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(window, f"X of object: {round(x, 2)} cm", (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(window, f"Y of object: {round(y, 2)} cm", (0, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            
            #cv.imshow("objcoords", window)
            cv.imshow("objcoords", cv.resize(window, (0,0), fx=0.5, fy=0.5))

        #draws a window that says "TRACKING LOST" if any of the variables aren't reassigned.
        elif coordsL is None or coordsR is None or left_y is None or filtered_disp_vis is None:
            window = np.zeros((500, 500, 3), np.uint8)
            cv.putText(window, "WARNING: TRACKING LOST!", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            
            #cv.imshow("objcoords", window)
            cv.imshow("objcoords", cv.resize(window, (0,0), fx=0.5, fy=0.5))

        #NOTE Uncomment this if you want to save each frame. You will need to create the 
        #directories as well.

        #cv.imwrite(f"src/merits_project/merits_project/depth_frame/depth{self.counter}.jpg", filtered_disp_vis)
        #cv.imwrite(f"src/merits_project/merits_project/camR/imgR{self.counter}.jpg", frame_r_show)
        #cv.imwrite(f"src/merits_project/merits_project/camL/imgL{self.counter}.jpg", frame_l_show)
        #cv.imwrite(f"src/merits_project/merits_project/window/coords{self.counter}.jpg", window)

        #self.counter += 1

        #NOTE without this, the images will not show up. keep it if you want to 
        #see real-time feedback
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()

#typical ROS2 node spinning main function.
def main(args=None):
    rclpy.init(args=args)
    node = CoordsCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    
