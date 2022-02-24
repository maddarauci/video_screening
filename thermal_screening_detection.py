import os 
import numpy as np
import cv2 

# NO TESTS HAVE BEEN DONE !!

base_dir = "assets"
threshold = 200
area_of_box = 700   # 300 for img input 
min_temp = 102      # 2 for img img 
font_scale_caution = 1 # 2 for img input
font_scale_temp = 0.7 # 1 for img input


def convert_to_temperature(pixel_avg):
    # converts pixel value (mean) to temperature (fahrenheit) depending upon the camera hardware.
    return pixel_avg / 2.25

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

    # Binary threshold.
    _, binary_threshold = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)

    # image opening: Erosion followed by dilation
    kernal = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernal, interations=1)
    image_opening = cv2.dilate(image_erosion, kernal, iterations=1)

    # Get contoures from the image obtained by opening operation
    contours, _ = cv2.findContoures(image_opening, 1, 2)

    image_with_rectangle = np.copy(heatmap)

    for contoure in contoures:
        # recatangle over each contoure
        x, y, w, h = cv2.boundingRect(contour)

        # Pass if the area of rectangle is not large enough.
        if (w) * (h) < area_of_box:
            continue


        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # mean of only those pixels which are in block and not the whole rectangle selected.
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Colors for rectangle and textmin_area
        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < min_temp else (
            255, 255, 127
        )
        # Callback function if the following condition is true.
        if temperature >= min_temp:
            # Call back function here 
            cv2.putText(image_with_rectangles, "High temperature detected!!", (35, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_detection, color, 2, cv2.LINE_AA)

        # Draw rectangles for visualation
        image_with_rectangles = cv2.rectangle(
            image_with_rectangles, "{} F".format(temperature), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_temp, color, 2, cv2.LINE_AA)
        )

    return image_with_rectangles

def main():
    # main driver function 

    # for Vidoe input
    Video = cv2.VideoCapture(str(base_dir+'video_input.mp4'))
    Video_frames = []

    while True:
        ret, frame = video.read()

        if not ret:
            break
        # process each frame
        frame = processs_frame(frame)
        height, width, _ = frame.shape
        Video_frame.append(frame)

        # Show the vidoe as it is being processed in a window
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    Video.release()
    cv2.destroyAllWindows()

    # save video to ouput 
    size = (height, width)
    out = cv2.VidoeWriter(str(base_dir+'output.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 100, size)

    for i in range(len(video_frames)):
        out.write(video_frames[i])
    out.release()

    # img = cv2.imread(str(base_dir + 'input_image.jpg))
    #
    # process the image 
    processed_img = process_frame(img)
    height, width, _ = processed_img.shape
    dim = (int(width * 0.5), int(height * 0.5))
    #
    # resize_img = cv2.resize(processed_img, dim, interpilation=cv2.INTER_AREA)
    # cv2.imwrite(str(base_dir + 'output_image.jpg), resize_img)
    #
    # saved_img = cv2.imread(str(base_dir + 'ouput_image.jpg'))
    # cv2.imshow('output', saved_img)
    #
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()