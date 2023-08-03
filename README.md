# 3D-printer-head-tracker
A program made for my MERITS summer internship. It tracks the coordinates of a 3D printer head in real time and publishes them in hopes of allowing the operator to catch Layer Shifting errors that take place during the printing process.

#
Program dependencies include ROS2 Foxy, OpenCV(Latest version works fine) and NumPy, and Ultralytics.
-

# How To Use
1. Make sure you have all the dependencies installed (see above).
2. Install repo as a directory in /home/<user> directory.
3. Source /opt/ros/foxy/setup.bash in /home/<user> directory.
4. Generate remapping files for your stereo camera setup using the files in remapping_creator. Use calib_img.py to save chessboard images, then use stereo_calibration.py to generate remapping file in the remapping_creator directory. Then, place that file in src/merits_project/merits_project/opencv_assets.
5. Change parameter declarations in the launch files to include the relative path for your new remapping file (parameter can be found in src/merits_project/merits_project/coords_synchronized.py, and can be changed in src/project_launch/launch/depth_tracker.launch.py or /depth_tracker_no_map.launch.py).
6. Change parameter declarations in the launch files to include the camera index number for your left and right cameras (parameter can be found in src/merits_project/merits_project/camera_node.py, and can be changed in stc/project_launch/launch/depth_tracker.launch.py or /depth_tracker_no_map.launch.py).
7. When I used this for my model, my stereo cameras were on a wooden block and had the lenses tilted 15 degrees upwards. If your setup does not mimic this, DELETE the y_add line in coords_synchronized.py. 
8. Test the program. After testing, change the parameters as necessary to shift zeros of the X, Y, and Z axis to your desired locations. It is likely you will also need to change the values to track the nozzle of the printer, as the program initially tracks the location of the center of the printer's face.
9. The YOLO file is trained to detect the front face of the Ender-3 3D printer head (the side that has the ender logo on it). If you want to track another 3D printer head, feel free to train a YOLOv8 model and put the .pt file in src/merits_project/merits_project/opencv_assets and change the YOLO file used in the coords_synchronized.py file accordingly. This is also a parameter that can be set in the launch files.
10. The program should run as intended now, displaying the two camera frames with bounding boxes around the 3D printer head, the disparity map (if you launched with the disparity map), and the window with the printer head coordinates on it.
11. As an extra, you can also change the Stereo_SGBM parameter if launching with the disparity map to determine whether you want to use semi-global block matching or regular block-matching to calculate the disparity map.

# Error reporting
I am relatively new to programming and would like to get any feedback on errors in ineffective practices in my code. Please message me if there are any of these errors so I can work on developing a more refined program. My email is henrymcd@yahoo.com. Thank you!

# Acknowledgements
I wouldn't have been able to do this project without multiple very important people working alongside me and offering me advice. I cannot repeat enough my profuse gratitude towards:
1. Vikas Dhiman, who accepted me into the lab and gave me a chance to grow my skills despite my inexperience. He bought the materials I needed for this project and gave me pointers on my code, how to approach problems and projects, and proper presentation formatting.
2. Dyllon Dunton and Jacob Wildes, who volunteered to mentor me and guided my project goals throughout the summer. They were frequently there to guide me through important programming processes (like uploading to github), and let me test my program in their apartment three times!
3. Everyone else in the lab, including Chris, Jeff, Masoud, Melis, and Shihab, who all contributed to my project through either advice or construction.
4. The Maine Space Grant Consortium for running the MERITS program and funding my stay at UMaine.
