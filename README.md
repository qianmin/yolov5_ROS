# yolov5_ROS
run YOLOv5 in ROS  

# requiremens
①an conda env called yolo,in which you can run yolov5 dependently  
②ROS kinetic, ubuntu16.04  
③basic usb_cam ROS_driver  

# command to run
```
1:roscore
2:rosrun usb_cam usb_cam  
3:yolo #enter into your conda env
4:rosrun ros_yolo final_detect.py  
```


