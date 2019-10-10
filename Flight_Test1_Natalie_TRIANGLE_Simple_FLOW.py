import cv2 as cv
import numpy as np
import dronekit as dk
import dronekit_sitl
from pymavlink import mavutil
import time
from time import gmtime, strftime
import os

def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = dk.VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def goto_position_target_local_ned(north, east, down):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111111000,  # type_mask (only positions enabled)
        north, east, down,  # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0,  # x, y, z velocity in m/s  (not used)
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    # send command to vehicle
    vehicle.send_mavlink(msg)
    vehicle.flush()

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        # if ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) ** 0.5 > 15:
        if ((x2 - x1) > 10 or (x2 - x1) < -10) and ((y2 - y1) > 10 or (y2 - y1) < -10):
            cv.circle(vis, (x1, y1), 15, (0, 0, 255), -1)
            cv.circle(vis, (x2, y2), 15, (0, 0, 255), -1)

    # cv.polylines(vis, lines, 0, (0, 255, 0))

    return vis
#END of definitions

connection_string = '/dev/ttyACM0'#, 115200' 	#CONNECTION TO PIXHAWK
vehicle = dk.connect(connection_string, wait_ready=True, baud=115200)

#RECORDING VIDEO SETUP
dir_original = 'ORIGINAL'
dir_opt_flow = 'OPT_FLOW'
time_stamp = strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out_original = cv.VideoWriter(os.path.join(dir_original,'original_'+time_stamp+'.avi'),fourcc, 8.0, (640,480)) #set file to write original camera input
out_opt_flow = cv.VideoWriter(os.path.join(dir_opt_flow, 'opt_flow'+time_stamp+'.avi'),fourcc, 8.0, (640,480)) #set file to write processed frames with optical flow
#out = cv.VideoWriter('output2.avi',fourcc, 8.0, (640,480))#changed FPS from 20 to 8

cmds = vehicle.commands
cmds.download()
cmds.wait_ready()

# init_lat = cmds[1].x #34.0531515
# init_lon = cmds[1].y #-114.67867319999999
waypoint1 = dk.LocationGlobalRelative(cmds[0].x, cmds[0].y, 3)  # Destination 1
waypoint2 = dk.LocationGlobalRelative(cmds[1].x, cmds[1].y, 3)  # Destination 1
arm_and_takeoff(3)
vehicle.airspeed = 0.5 # set drone speed to use with simple_goto
vehicle.simple_goto(waypoint1)#trying to reach 1st waypoint
time.sleep(20)

cam = cv.VideoCapture(0)
ret, prev = cam.read()
h, w = prev.shape[:2]
prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

while True:
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.6, 5, 15, 3, 5, 1.2, 0)
    prevgray = gray
    new_frame = draw_flow(gray, flow)
    threshold = 10 # !!!!!!!!!!!!!!!! T H R E S H O L D

    lyL, lxL = np.mgrid[1:h, 1:w / 2].reshape(2, -1).astype(int)  #
    ldxL, ldyL = flow[lyL, lxL].T
    lyR, lxR = np.mgrid[1:h, w/2:w].reshape(2, -1).astype(int)  #
    ldxR, ldyR = flow[lyR, lxR].T 
    if max(ldxL) > threshold:
        print("turn right")
        send_ned_velocity(0, 0, 0)  # stop the vehicle for 5 seconds
        time.sleep(2)
        goto_position_target_local_ned(0, -2, 0)  # move right for 2 meters
        time.sleep(4)
    elif min(ldxR) < threshold*(-1):
        print("turn left")
        send_ned_velocity(0, 0, 0)  # stop the vehicle for 5 seconds
        time.sleep(2)
        goto_position_target_local_ned(0, -2, 0)  # move right for 2 meters
        time.sleep(4)
    print("go to destination 1 sec") 
    vehicle.simple_goto(waypoint2)
    time.sleep(1)

    out_original.write(img)
    out_opt_flow.write(new_frame)
    #cv.imshow("OpticalFlow", new_frame) 
    #cv.imshow("Original", frame_gray)
    

    key = cv.waitKey(30)
    if key == ord('q'):
        out_opt_flow.release()
        out_original.release()
        break
    lat = vehicle.location.global_relative_frame.lat  # get the current latitude
    lon = vehicle.location.global_relative_frame.lon  # get the current longitude
    if lat == cmds[0].x and lon == cmds[0].y:  # check whether the vehicle is arrived or not
        print("Arrived")
        out_opt_flow.release()
        out_original.release()
        break

print("Landing")
vehicle.mode = dk.VehicleMode("LAND")
vehicle.flush()
