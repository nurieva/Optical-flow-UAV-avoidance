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


#RECORDING VIDEO SETUP
dir_original = 'ORIGINAL'
dir_opt_flow = 'OPT_FLOW'
time_stamp = strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out_original = cv.VideoWriter(os.path.join(dir_original,'original_'+time_stamp+'.avi'),fourcc, 8.0, (640,480)) #set file to write original camera input
out_opt_flow = cv.VideoWriter(os.path.join(dir_opt_flow, 'opt_flow'+time_stamp+'.avi'),fourcc, 8.0, (640,480)) #set file to write processed frames with optical flow
#out = cv.VideoWriter('output2.avi',fourcc, 8.0, (640,480))#changed FPS from 20 to 8


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
    ly, lx = np.mgrid[1:h, 1:w / 2].reshape(2, -1).astype(int)  #
    ldx, ldy = flow[ly, lx].T
    if max(ldx) > threshold:
        print("turn right")
        
    ly, lx = np.mgrid[1:h, w/2:w].reshape(2, -1).astype(int)  #
    ldx, ldy = flow[ly, lx].T    
    if min(ldx) < threshold*(-1):
        print("turn left") 

    out_original.write(img)
    out_opt_flow.write(new_frame)
    cv.imshow("OpticalFlow", new_frame) 
    cv.imshow("Original", img)
    

    key = cv.waitKey(30)
    if key == ord('q'):
        out_opt_flow.release()
        out_original.release()
        break
    
print("Landing")

