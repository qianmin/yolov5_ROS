# yolov5_ROS
this ros package can run YOLOv5 in ROS 

# requirements
①an conda env called yolo,in which you can run yolov5 dependently  
②ROS kinetic, ubuntu16.04  
③basic usb_cam ROS_driver  

# command to run
**copy this package to your catkin_ws/src
and then catkin_make
(in fact ,this python paakage dont need to build at all)
```
1:roscore
2:rosrun usb_cam usb_cam  
3:yolo                               #enter into your conda env
4:rosrun ros_yolo final_detect.py  
```


