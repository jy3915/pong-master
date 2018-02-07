from collections import deque
import numpy as np
import imutils
import cv2

# Settings
winWidth = 800
winHeight = 600
timeDelta = 2
# Image processing setting
Blurred = True
backgroundSubtraction = True
noiseReduction = True
# Define HSV color bound
hsvColorBounds = {}
hsvColorBounds['green'] = (np.array([29, 86, 6]), np.array([64, 255, 255]))
hsvColorBounds['red'] = (np.array([6, 29, 86]), np.array([64, 64, 255]))
hsvColorBounds['pink'] = (np.array([100, 90, 200]),np.array([180, 120, 255]))
# Currently in use color bound
colorBounds = hsvColorBounds['pink']

# Frame capture and window resize 800x600
def getFrame(camera):
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=winWidth, height=winHeight)
    return frame
# Frame thresholding
def frameThresholding(frame):
    # Blurring the frame
    if Blurred:
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
    # Subtract background (makes isolation of balls more effective, in combination with thresholding)
    if backgroundSubtraction:
        fgbg = cv2.createBackgroundSubtractorMOG2()
        fgmask = fgbg.apply(frame)
        frame = cv2.bitwise_and(frame,frame, mask = fgmask)
    if noiseReduction:
        kernel = np.ones((3, 3)).astype(np.uint8)
        frame = cv2.erode(frame, kernel)
        frame = cv2.dilate(frame, kernel)
    # Convert to HSV color space
    hsvBlurredFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsvBlurredFrame
# Ball bounce at boundaries
def boundaryBounce(y):
    if y > winHeight:
        y = 2 * winHeight - y
    elif y < 0:
        y = -y
    return y
# Main
def main():
    # Init data structure
    ballCenters = deque(maxlen=16)
    ballVelocities = deque(maxlen=16)
    ballCenterTraject = deque(maxlen=16)
    counter = 0
    # Define video source
    camera = cv2.VideoCapture(0)
    # Main Loop
    while True:
        # Grab current frame
        frame = getFrame(camera)
        # Thresholding
        hsv = frameThresholding(frame)
        # Construct a mask for colors
        mask = cv2.inRange(hsv, colorBounds[0], colorBounds[1])
        # Check masked view
        cv2.imshow('Mask',mask)
        # Find contours in the mask
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        # Init center coordinates
        center = None
        velocity = None
        nextCenter = None

        # Find center coordinate,only proceed if at least one contour was found
        if len(contours) > 0:
            # Find the largest contour in the mask
            contourMax = max(contours, key=cv2.contourArea)
            # Compute minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(contourMax)
            M = cv2.moments(contourMax)
            if int(M["m00"])!= 0:
                posX = int(M["m10"] / M["m00"])
                posY = int(M["m01"] / M["m00"])
                center = (posX,posY)
            # Only proceed if the radius meets a minimum size
            if radius > 10:
                # Draw the circle and centroid
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # Update the center queue
        ballCenters.appendleft(center)

        # Compute ball velocity
        for i in np.arange(1, len(ballCenters)):
            # If either of the tracked points are None, ignore
            if ballCenters[i - 1] is None or ballCenters[i] is None:
                continue
            # If enough points in the buffer
            if counter >= 10 and i == 1 and ballCenters[-10] is not None:
                # compute dX/df and dY/df  df = change in frame
                dX = ballCenters[i][0] - ballCenters[-10][0]
                dY = ballCenters[i][1] - ballCenters[-10][1]
                velocity = (dX,dY)
        # Update the velocity queue
        ballVelocities.appendleft(velocity)

        # Compute coordinate of predicted position in timedelta
        for i in np.arange(1, len(ballVelocities)):
            # If either of the tracked points are None, ignore
            if ballVelocities[i - 1] is None or ballVelocities[i] is None:
                continue
            # If enough points in the buffer
            if counter >= 10 and i == 1 and ballVelocities[-10] is not None:
                nextPositionX = ballCenters[i][0] + ballVelocities[i][0] * timeDelta  # x velocity
                nextPositionY = ballCenters[i][1] + ballVelocities[i][1] * timeDelta  # y velocity
                # Bounce at boundary
                nextPositionY = boundaryBounce(nextPositionY)
                nextCenter = (nextPositionX,nextPositionY)
                cv2.line(frame, nextCenter, center, (0, 0, 255), 5)
        # Update the next center queue
        ballCenterTraject.appendleft(nextCenter)

        # Add 1 counter per loop
        counter += 1

        # Show frame window
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to exit
        if key == ord("q"):
            break

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

############################### MILESTONE ACHIEVED#############################

# Able to work out position(i+1) (predicted position), draw predicted trajectory
# Rewrote some cv2 based image processing to functions, easier to call (really?)
# IT BOUNCE!!!
# Supper short if shrink all functions :)


# Next step: Based on FPS to work out pixel position change per second (dX/dt,dY/dt).
#           Adapt the vision of the camera for meters per pixel (MPP).
#           Workout the actual position (in meters) and velocity (in m/s)


# R = FPS * mm per pixel
# ds/dt = dx/df * R