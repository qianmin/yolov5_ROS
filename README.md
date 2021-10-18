# yolov5_ROS
this is and successfull ros package，which can run YOLOv5 in ROS 
### (只要你有一个能跑yolov5的python环境，那么就可以在ros里面跑yolov5！，和ros版本无关！，和ubuntu系统无关）

# 1:requirements
①an conda env called yolo,in which you can run yolov5 dependently  

②any ROS,any Ubuntu 

③basic usb_cam ROS_driver  (ros自带）

④yolov5s.pth or else ***(if you dont't have it,it will automatically donwload to your system)***

⑤pip install rospy  (安装ros与python的接口，这就是为什么无关的原因）

# 2:before run
```
1:copy this package(ros_yolo) to your catkin_ws/src  
2:catkin_make  
3:in final_yolo.py, you need to change the image_topic to your own camera input topic  
```

# 3:command to run
```
1:roscore
2:rosrun usb_cam usb_cam_node                       #启动ros自带的相机功能包 ros自带
3:conda activate yolo                               #进入你的conda环境
4:rosrun ros_yolo final_yolo.py  
```
# 4:results
![yolo](./readme/yolo.png)


