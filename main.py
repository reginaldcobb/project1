import os
import sys
import time
import socket
import json
import cv2
import math

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. image files should end with .jpg or .bmp.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def get_center(xmin, xmax, ymin, ymax):
    """
    Determines the centroid of a tracking box.

    :param xmin, ymin:  The upper top left corner of the tracking box
    :param xmax, ymax:  The bottom right corner of the tracking box
    :return cX, cY:  The centroid X and Y points for the tracking box
    """
    cX = int((xmin + xmax) / 2.0)
    cY = int((ymin + ymax) / 2.0)
    return cX, cY

def get_distance(cX, cY, cX1, cY1):
    """
    Determine Distance between two Centroid points.

    :param cX, cY:  The centroid of the current tracking box
    :param cX1, cY1:  The centroid of the previous tracking box
    :return dist:  The distance between the two centroid points 
    """
    dist = 0.0
    dX = cX - cX1
    dY = cY - cY1
    dist = math.sqrt( math.pow(dX, 2) + math.pow(dY, 2))            
    return dist

def ssd_out(frame, result, distance_thres,  start_video, prev_xmin, prev_cX, prev_cY):
    """
    Parse SSD output.

    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :param distance_thres: Threshold for check centroids to determine if same person still on screen
    :param   start_video: Threshold for check centroids to determine if same person still on screen
    :param   prev_xmin: Previous minimum X value for box
    :param   prev_cX: Previous's frame centroid X coordinate
    :param   prev_cY: Previous's frame centroid Y coordinate
    :return: frame,  people_in, prev_xmin, prev_cX, prev_cY,  start_of_person
    """
    dist = 0.0
    people_in = 0
    # stop_counter = 0
    # new_person = False
    start_of_person = 0

    for obj in result[0][0]:  # Draw bounding box for object when it's probability is more than the specified threshold     
        if (obj[2] >= prob_threshold):
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)

            # Add the model class and probability to the box
            label = classNames[obj[1]] + ": " + str(int(obj[2]*100)) + "%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, label, (xmax, ymin + labelSize[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255))
            
            # Draw the tracking box to the fram
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax ), (255, 255, 255))

            # Display center dot within box
            cX, cY = get_center( xmin, xmax, ymin, ymax)
            cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

            # check if this is the start of the video, if so, set the previous centroid values to Zero
            if (start_video):
                prev_cX = 0
                prev_cY = 0

            # Check to see if the image is coming onto the screen.  If so, then this is the start of this person on the screen and a person is in the fram
            if (prev_xmin > 550) and (xmin < 325):
                start_of_person = 1
                people_in = 1

            #  Store the previous upper left box x-coord to check for new person entering the screen
            prev_xmin = xmin            
            
            # Get distance from previous centroid
            dist = get_distance( cX, cY, prev_cX, prev_cY)

            # Display previous center dot within box
            cX, cY = get_center( xmin, xmax, ymin, ymax)
            cv2.circle(frame, (prev_cX, prev_cY), 4, (255, 0, 0), -1)

            # Update the previous centroid
            prev_cX = cX
            prev_cY = cY

            #  If the tracking box is bouncing around because of model noise, then ignore and record as person in frame
            if  (dist < distance_thres):
                people_in = 1
            # If tracking box has moved to far, then record that person has left the frame
            if (dist > distance_thres):
                people_in = 0

        else:
            # If a person was detected by model but it didn't pass the prob threshold test, then set people in flag to 0 to display in UI
            if (obj[1] == 15):
                people_in = 0

    return frame, people_in, prev_xmin, prev_cX, prev_cY,  start_of_person

def main():
    """
    This is the main processing model of this python script

    Load the network and parse the SSD output.
    :return: None
    """
    # Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    args = build_argparser().parse_args()

    # Initialize value for request ID
    request_id = 0

    # Flag for the input image
    single_image_mode = False

    # last_count = 0
    prev_cX = 0
    prev_cY = 0
    prev_xmin = 767 # Set a threshold to check for people leaving the right edge of the frame

    start_video = True
    timer_started = False
    start_time = 0    

    last_start_of_person = 0
    prev_people_in = 0
    
# Initialize the class
    infer_network = Network()

    # Check for Video, Image or other imputs
    if args.input == 'CAM':
        input_device = 0 # Setting to zero for cv2 will read from the webcam connected to the computer

    # Checks for video file
    elif (args.input.endswith('.jpg') or args.input.endswith('.bmp')) and (os.path.isfile(args.input)):
        single_image_mode = True
        input_device = args.input

    # Checks for video file
    elif (os.path.isfile(args.input)):
        input_device  = args.input

    else:        
        log.error("ERROR! Unable to open requested input source ")
        log.error("Exiting ...")
        exit(1)

    infer_network.load_model(args.model, args.device, args.cpu_extension, request_id)
    net_input_shape = infer_network.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(input_device)
    cap.open(input_device)

    global initial_w, initial_h, prob_threshold

    # Get the threshold arg
    prob_threshold = args.prob_threshold

    # Get the width and height of the frame to be processed
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # Process frames until the video ends, or process is exited

    while cap.isOpened():
        # total_frames = total_frames + 1

        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Start asynchronous inference for specified request.
        inf_start = time.time()

        # Perform inference on the frame
        infer_network.async_inference(p_frame,request_id)

        # Get the output of inference
        if infer_network.wait(request_id) == 0:
            det_time = time.time() - inf_start
            result = infer_network.get_output(request_id)

            # if first person, then start the timer
            if timer_started == False:
                start_time = time.time()

            # Process the image
            frame,  people_in, prev_xmin,  prev_cX, prev_cY, start_of_person = ssd_out(frame, result, 300,  start_video, prev_xmin, prev_cX, prev_cY)

            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 10, 10), 1)

            # Ccheck if person is off screen
            if (people_in == 1) and (prev_people_in == 1):
                off_screen = time.time()

            # Check if the person has returned from off screen
            if (people_in == 0) and (prev_people_in == 1):
                off_screen = int(time.time() - off_screen )

            # If person has gone off screen, person duration in the video is calculated
            if (start_of_person == 1) and (last_start_of_person == 0) and (timer_started == True):
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))   

            #  If person has just come on screen, then start a timer
            if (start_of_person == 0) and (last_start_of_person == 1):
                start_time = time.time()
                timer_started = True

            client.publish("person", json.dumps({"count": people_in})) #People Detected In Frame

            # Save state variables
            last_start_of_person = start_of_person
            prev_people_in = people_in

            # Break if escape key pressed
            if key_pressed == 27:
                break
        
        # Send frame to the ffmpeg server
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        # Write the output image if the input was an image
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
        
        #  Set starting frame flag to fales since the first fram has now been processes
        start_video = False

    # write the final onscreen time
    duration = int(time.time() - start_time)
    client.publish("person/duration", json.dumps({"duration": duration}))   

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


if __name__ == "__main__":
    main()
