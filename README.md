# yolov5_ROS
run YOLOv5 in ROS  

# requiremens
①an conda env called yolo,in which you can run yolov5 dependntly  
②ROS kinetic, ubuntu16.04  
③basic usb_cam ROS_driver  

## run
``
roscore
rosrun usb_cam usb_cam  

rosrun ros_yolo final_detect.py  

``
