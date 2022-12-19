'''
    Author: Bharadwaj Chukkala
    UID: 118341705
'''

#### Importing Libraries
#
#
import cv2 as cv  # OpenCV Library
import numpy as np  # Numpy Library

'''
Function to perform Image Warping to view the lane lines from a Bird's eye perspective
'''
def warp_image(img):
    dst = np.array([(500, 300), (0, 300), (0, 0), (500, 0)], np.float32)
    src = np.array([(1100, 660), (200, 680), (600, 450), (730, 445)], np.float32)

    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, (500, 300), flags=cv.INTER_NEAREST)

    return warped

'''
Function to perform Inverse warping to project lane lines back in original perspective
'''
def unwarp_image(img):
    dst = np.array([(500, 300), (0, 300), (0, 0), (500, 0)], np.float32)
    src = np.array([(1100, 660), (200, 680), (600, 450), (730, 445)], np.float32)

    img_size = (img.shape[1], img.shape[0])
    Minv = cv.getPerspectiveTransform(dst, src)
    unwarped = cv.warpPerspective(img, Minv, img_size, flags=cv.INTER_NEAREST)

    return unwarped

'''
Function to Detect image edges using Canny edge Detector
'''
def detect_edges(image):
    # RGB to HLS
    hls_image = cv.cvtColor(image, cv.COLOR_RGB2HLS)

    # Getting Saturation Channel
    s_channel = hls_image[:, :, 2]

    # Performing Canny Edge Detection
    image_edges = cv.Canny(s_channel, 100, 200)

    # Thresholding to remove Errors
    ret, thresh_image = cv.threshold(image_edges, 120, 255, cv.THRESH_BINARY)

    return thresh_image, s_channel, image_edges

'''
Function to create a region of interest and perform masking operation
'''
def mask(edge_image):
    # Extract Dimensions
    img_height, img_width = edge_image.shape[:2]

    # Slicing the region of interest
    # Boundary points of roi trapeziod
    roi = np.array([[(150, img_height), (620, 420), (750, 420), (1200, img_height)]])  

    # Creating Mask using fillpoly
    blank_mask = np.zeros_like(edge_image)  # Blank image with frame dimensions
    mask = cv.fillPoly(blank_mask, roi, 255)
    mask = mask.astype(np.uint8)
    output_image = cv.bitwise_and(edge_image, mask, mask=None)
    return output_image


'''
Function to perform Polynomial Curve Fitting using a Sliding Window 
'''
def Sliding_window(warped_img):
    histogram = np.sum(warped_img, axis=0)

    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

    midpoint = int(histogram.shape[0] / 2)
    leftlanepixel_initial = np.argmax(histogram[:midpoint])
    rightlanepixel_initial = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftlanepixel_current = leftlanepixel_initial
    rightlanepixel_current = rightlanepixel_initial

    image_center = int(warped_img.shape[1] / 2)


    left_lane_idxs = []
    right_lane_idxs = []

    window_height = int(warped_img.shape[0] / num_windows)

    for window in range(num_windows):

        win_y_down = warped_img.shape[0] - (window + 1) * window_height
        win_y_up = warped_img.shape[0] - window * window_height
        win_x_left_down = leftlanepixel_current - window_width
        win_x_right_down = rightlanepixel_current - window_width
        win_x_left_up = leftlanepixel_current + window_width
        win_x_right_up = rightlanepixel_current + window_width

        cv.rectangle(out_img, (win_x_left_down, win_y_down), (win_x_left_up, win_y_up), (120, 120 , 120), 1)
        cv.rectangle(out_img, (win_x_right_down, win_y_down), (win_x_right_up, win_y_up), (120, 120, 120), 1)

        good_left_idxs = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (
                nonzerox < win_x_left_up)).nonzero()[0]
        good_right_idxs = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (
                nonzerox < win_x_right_up)).nonzero()[0]

        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)

        if len(good_left_idxs) > minpix:
            leftlanepixel_current = int(np.mean(nonzerox[good_left_idxs]))
        if len(good_right_idxs) > minpix:
            rightlanepixel_current = int(np.mean(nonzerox[good_right_idxs]))

    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    left_pixels_x = nonzerox[left_lane_idxs]
    left_pixels_y = nonzeroy[left_lane_idxs]
    right_pixels_x = nonzerox[right_lane_idxs]
    right_pixels_y = nonzeroy[right_lane_idxs]

    out_img[nonzeroy[right_lane_idxs], nonzerox[right_lane_idxs]] = [255, 0, 0]
    out_img[nonzeroy[left_lane_idxs], nonzerox[left_lane_idxs]] = [0, 0, 255]

    left_fit = np.polyfit(left_pixels_y, left_pixels_x, 2)
    right_fit = np.polyfit(right_pixels_y, right_pixels_x, 2)

    left_fit_avg.append(left_fit)
    right_fit_avg.append(right_fit)

    ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((left_line_pts, right_line_pts))
    pts = np.array(pts, dtype=np.int32)

    color_blend = np.zeros_like(original_image).astype(np.uint8)
    cv.fillPoly(color_blend, pts, (0, 0, 120))

    Unwarped_img = unwarp_image(color_blend)
    result = cv.addWeighted(original_image, 1, Unwarped_img, 0.5, 0)

    return result, out_img, left_fit, right_fit

'''
Function to calculate the radius of curvature of the lanes
'''
def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(ploty * ymtr_per_pixel, left_fitx * xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty * ymtr_per_pixel, right_fitx * xmtr_per_pixel, 2)

    left_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ymtr_per_pixel + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_rad = ((1 + (2 * right_fit_cr[0] * y_eval * ymtr_per_pixel + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return (left_rad, right_rad)

'''
Function to display curvature and plot the calculations as text on the video
'''
def show_curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)

    if (left_curvature > right_curvature):
        prediction = "Right turn ahead"
    elif (left_curvature == right_curvature):
        prediction = "Straight road"
    else:
        prediction = "Left turn ahead"

    avg_rad = round(np.mean([left_curvature, right_curvature]), 0)

    cv.putText(img, 'Average lane curvature: {:.2f} m'.format(avg_rad),
               (50, 50), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    cv.putText(img, 'left lane curvature: {:.2f} m'.format(left_curvature),
               (50, 80), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    cv.putText(img, 'right lane curvature: {:.2f} m'.format(right_curvature),
               (50, 110), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    cv.putText(img, prediction, (50, 140), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0 ,0), 2)

    return img

'''
Function to calculate the car position in the image
'''
def car_position(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ymax = img.shape[0] * ymtr_per_pixel

    center = img.shape[1] / 2

    lineLeft = left_fit[0] * ymax ** 2 + left_fit[1] * ymax + left_fit[2]
    lineRight = right_fit[0] * ymax ** 2 + right_fit[1] * ymax + right_fit[2]

    mid = lineLeft + (lineRight - lineLeft) / 2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0.:
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))

    return message

'''
Function to display the final outputs
'''
def display_process(org_img, edges_img, bird_img, sliding_img, final_output):

    # Output Window Dimensions
    height, width = 1080, 1920

    # Output Window
    final_img = np.zeros((height, width, 3), np.uint8)

    # Creating depth for Edges and Warped Image
    edges_img = np.dstack((edges_img, edges_img, edges_img))
    bird_img = np.dstack((bird_img, bird_img, bird_img))

    # Text
    cv.putText(org_img, '[1] Undistorted Image', (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, 0)
    cv.putText(edges_img, '[2] Detected Lane Lines (Canny)', (30, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, 0)
    cv.putText(bird_img, '[3] Warped Image', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, 0)
    cv.putText(sliding_img, '[4] Polynomial Curve Fitting', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, 0)

    # Predicted Image
    final_img[0:620, 640:1920] = cv.resize(final_output, (1280, 620), interpolation=cv.INTER_AREA)

    # Original Image
    final_img[0:620, 0:640] = cv.resize(org_img, (640, 620), interpolation=cv.INTER_AREA)

    # Edge Image
    final_img[620:1080, 0:640] = cv.resize(edges_img, (640, 460), interpolation=cv.INTER_AREA)

    # Warped Image
    final_img[620:1080, 640:1280] = cv.resize(bird_img, (640, 460), interpolation=cv.INTER_AREA)

    # Sliding Window
    final_img[620:1080, 1280:1920] = cv.resize(sliding_img, (640, 460), interpolation=cv.INTER_AREA)

    neon = np.zeros((100, final_img.shape[1], 3), np.uint8)
    neon[:] = (255, 0, 180)

    final_img = cv.vconcat((final_img, neon))

    return final_img


w, h = 1280, 720

num_windows = 10
window_width = 50
minpix = 25

xmtr_per_pixel = 3 / 1280
ymtr_per_pixel = 30 / 720

left_fit_avg = []
right_fit_avg = []

# Object for reading Video
cap = cv.VideoCapture('challenge.mp4')

while cap.isOpened():
    ret, original_image = cap.read()
    if not ret:
        break

    threshold_image, saturation_channel_image, contour_detected_image = detect_edges(original_image)

    warped_img = warp_image(threshold_image)

    result, out_img, left_fit, right_fit = Sliding_window(warped_img)

    result = show_curvatures(result, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    message = car_position(original_image, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    cv.putText(result, message, (50, 170), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    masked_image = mask(contour_detected_image)
    
    turn_prediction = out_img


    # Showing all the outputs in one Window
    concatenated_image = display_process(original_image, masked_image, warped_img, out_img, result)
    cv.imshow('Complete Pipeline', concatenated_image)
    
    # frame_count = 1
    # if frame_count == 1:
    #     # print(os.listdir(directory))
    #     output1 = 'original_image.jpg'
    #     output2 = 'saturated_image.jpg'
    #     output3 = 'contour_detection.jpg'
    #     output4 = 'masked_image.jpg'
    #     output5 = 'warped_image.jpg'
    #     output6 = 'turn_prediction.jpg'
    #     output7 = 'result.jpg'
    #     output8 = 'concatenated_image.jpg'
    #     cv.imwrite(output1, original_image)
    #     cv.imwrite(output2, saturation_channel_image)
    #     cv.imwrite(output3, contour_detected_image)
    #     cv.imwrite(output4, masked_image)
    #     cv.imwrite(output5, warped_img)
    #     cv.imwrite(output6, turn_prediction)
    #     cv.imwrite(output7, result)
    #     cv.imwrite(output8, concatenated_image)
        
    # frame_count += 1

    # Wait key to visualize output
    if cv.waitKey(100) & 0xFF == ord('s'):
        break

# Closing Windows
cap.release()
cv.destroyAllWindows()