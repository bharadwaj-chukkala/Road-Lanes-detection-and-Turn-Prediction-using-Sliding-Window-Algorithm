'''
Author: Bharadwaj Chukkala
UID: 118341705
'''

# Importing Necessary Libraries
import numpy as np
import cv2


'''
Function to define the region of interest
''' 
def region_of_interest(img):
    img_shape = img.shape[:2]
    vertices = np.array([[(0,img_shape[0]), (9*img_shape[1]/20, 11*img_shape[0]/18), (11*img_shape[1]/20, 11*img_shape[0]/18), (img_shape[1],img_shape[0])]], dtype=np.int32)
    mask = np.zeros_like(img).astype(np.uint8)
    cv2.fillPoly(mask, [vertices], (255,255,255))
    image= cv2.bitwise_and(mask,img)
    return image


'''
Function to extract lane Lines using Hough Transform
'''
def Hough_transform(image):
    
    img_shape = image.shape
    vertices = np.array([[(0,img_shape[0]), (9*img_shape[1]/20, 11*img_shape[0]/18), (11*img_shape[1]/20, 11*img_shape[0]/18), (img_shape[1],img_shape[0])]], dtype=np.int32)
   
    # Converting image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying a gaussian blur mask on the gray image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # detecting edges in the image
    edges = cv2.Canny(blur, 25, 100)

    # Defining the region  of interest
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(edges, mask)
    img_r_2=region_of_interest(image)
    new_img=cv2.bitwise_and(img_r_2,img_r_2,mask=masked)
    new_img=cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    
    # Performing Hough transform to detect lanes
    lines = cv2.HoughLinesP(new_img, 1, np.pi/180, 14, np.array([]), minLineLength=30, maxLineGap=60)
    hough_image = np.zeros((*new_img.shape, 3), dtype=np.uint8)

    return hough_image,masked,lines,edges    


'''
Function to get Histogram of the image
'''
def show_histogram(image):
    histogram=np.sum(image[image.shape[0]//2:,:],axis=0)
    return histogram


'''    
Function differentiates between the solid and dashed lines using Histogram peaks
'''
def lane_detector(image):
    img_hist= show_histogram(image)
    midpoint_current=int(img_hist.shape[0]/2)
    right_current_x=np.argmax(img_hist[midpoint_current:])+midpoint_current
    left_current_x=np.argmax(img_hist[:midpoint_current])
    non_zero_pixels=img_hist.nonzero()
    mid_reg=int(image.shape[1]/2)
    left_part=image[:,:mid_reg]
    right_part=image[:,mid_reg:]
    left_count = cv2.findNonZero(left_part)
    right_count=cv2.findNonZero(right_part)
 

    return left_current_x,right_current_x,non_zero_pixels,left_count,right_count

'''
Function to show output image
'''
def resultant(image,hough_image,lines,l_count,r_count):
   
    if  r_count.shape[0] > l_count.shape[0]:
        color_left = (0,0,255)
        text1="Left Lane: Dashed Lines Detected"
        color_right= (0,255,0)
        text2="Right Lane: Solid Lines Detected "
    else:
         print("left")
         color_left = (0,255,0)   
         text1 = "Left Lane: Solid Lines Detected "
         color_right = (0,0,255)    
         text2="Right Lane: Dashed Lines Detected " 
         
    # Plotting the left and right lane lines    
    # WEIGHTED IMAGE
    draw_lines(hough_image,lines,color_left,color_right)
    
    processed = cv2.addWeighted(image, 0.8, hough_image, 1, 0)
    
    cv2.putText(processed,text1,(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)
    cv2.putText(processed,text2,(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)

    
    return processed


'''
Function to draw different colored lines on the detected lane lines
'''
def draw_lines(img, lines,color_left,color_right,thickness=12):
   
    global CACHE_LEFT_SLOPE
    global CACHE_RIGHT_SLOPE
    global CACHE_LEFT
    global CACHE_RIGHT

    # DECLARE VARIABLES
    cache_weight = 0.9

    right_ys = []
    right_xs = []
    right_slopes = []

    left_ys = []
    left_xs = []
    left_slopes = []

    midpoint = img.shape[1] / 2

    bottom_of_image = img.shape[0]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope, yint = np.polyfit((x1, x2), (y1, y2), 1)
            # Filter lines using slope and x position
            if .35 < np.absolute(slope) <= .85:
                if slope > 0 and x1 > midpoint and x2 > midpoint:
                    right_ys.append(y1)
                    right_ys.append(y2)
                    right_xs.append(x1)
                    right_xs.append(x2)
                    right_slopes.append(slope)
                elif slope < 0 and x1 < midpoint and x2 < midpoint:
                    left_ys.append(y1)
                    left_ys.append(y2)
                    left_xs.append(x1)
                    left_xs.append(x2)
                    left_slopes.append(slope)
   
    
    # DRAW RIGHT LANE LINE
    if right_ys:
        right_index = right_ys.index(min(right_ys))
        right_x1 = right_xs[right_index]
        right_y1 = right_ys[right_index]
        right_slope = np.median(right_slopes)
        if CACHE_RIGHT_SLOPE != 0:
            right_slope = right_slope + (CACHE_RIGHT_SLOPE - right_slope) * cache_weight

        right_x2 = int(right_x1 + (bottom_of_image - right_y1) / right_slope)

        if CACHE_RIGHT_SLOPE != 0:
            right_x1 = int(right_x1 + (CACHE_RIGHT[0] - right_x1) * cache_weight)
            right_y1 = int(right_y1 + (CACHE_RIGHT[1] - right_y1) * cache_weight)
            right_x2 = int(right_x2 + (CACHE_RIGHT[2] - right_x2) * cache_weight)

        CACHE_RIGHT_SLOPE = right_slope
        CACHE_RIGHT = [right_x1, right_y1, right_x2]
        cv2.line(img, (right_x1, right_y1), (right_x2, bottom_of_image), color_right, thickness)
        
        

    # DRAW LEFT LANE LINE
    if left_ys:
        left_index = left_ys.index(min(left_ys))
        left_x1 = left_xs[left_index]
        left_y1 = left_ys[left_index]
        left_slope = np.median(left_slopes)
        if CACHE_LEFT_SLOPE != 0:
            left_slope = left_slope + (CACHE_LEFT_SLOPE - left_slope) * cache_weight

        left_x2 = int(left_x1 + (bottom_of_image - left_y1) / left_slope)

        if CACHE_LEFT_SLOPE != 0:
            left_x1 = int(left_x1 + (CACHE_LEFT[0] - left_x1) * cache_weight)
            left_y1 = int(left_y1 + (CACHE_LEFT[1] - left_y1) * cache_weight)

        CACHE_LEFT_SLOPE = left_slope
        CACHE_LEFT = [left_x1, left_y1, left_x2]
        cv2.line(img, (left_x1, left_y1), (left_x2, bottom_of_image), color_left, thickness)
        



def info_display(org_img, edges_img, bird_img, warp_img, final_output):  # , warped):

    # Output Window Dimensions
    height, width = 1080, 1920

    # Output Window
    final_img = np.zeros((height, width, 3), np.uint8)

    # Creating depth for Edges and Warped Image
    edges_img = np.dstack((edges_img, edges_img, edges_img))
    bird_img = np.dstack((bird_img, bird_img, bird_img))
    warp_img = np.dstack((warp_img, warp_img, warp_img))

    # Text
    cv2.putText(org_img, '[1] Input Image Frame', (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 3, 0)
    cv2.putText(edges_img, '[2] Detected Contours', (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 3, 0)
    cv2.putText(bird_img, '[3] Masked Image', (60, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 3, 0)
    cv2.putText(warp_img, '[4] Warpped Image', (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, 0)

    # Predicted Image
    final_img[0:620, 640:1920] = cv2.resize(final_output, (1280, 620), interpolation=cv2.INTER_AREA)

    # Original Image
    final_img[0:620, 0:640] = cv2.resize(org_img, (640, 620), interpolation=cv2.INTER_AREA)

    # Edge Image
    final_img[620:1080, 0:640] = cv2.resize(edges_img, (640, 460), interpolation=cv2.INTER_AREA)

    # Warped Image
    final_img[620:1080, 640:1280] = cv2.resize(bird_img, (640, 460), interpolation=cv2.INTER_AREA)

    # Sliding Window
    final_img[620:1080, 1280:1920] = cv2.resize(warp_img, (640, 460), interpolation=cv2.INTER_AREA)

    neon = np.zeros((100, final_img.shape[1], 3), np.uint8)
    neon[:] = (255, 0, 180)

    final_img = cv2.vconcat((final_img, neon))

    return final_img

# Source points for homography.
bird_eye_coords_= np.float32([[410,335], [535, 334], [780, 479], [150, 496]])
# bird_eye_coords_=np.float32([[422,321],[540,330],[790,485],[80,500]])
# Destination points for homography
world_coords_ = np.float32([[50, 0], [250, 0], [250, 500], [0, 500]])


if __name__ == "__main__":
    source = 'whiteline.mp4'
    cap = cv2.VideoCapture(source)
    print('Reading Video ...')
    print("Detecting Lanes in each frame ...")
    output_filename='output_videos/output_video2.mp4'
    size=(960,540)
    
    CACHE_LEFT_SLOPE = 0
    CACHE_RIGHT_SLOPE = 0
    CACHE_LEFT = [0, 0, 0]
    CACHE_RIGHT = [0, 0, 0]
    
    result = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    Frame=0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            Frame+=1
            print('Frame: ',Frame) 
            
            # Uncomment the below line to see the video flipped horizontally
            #img=cv2.flip(img,1) 
            
            hough_image,masked_edges,lines,edges=Hough_transform(img)
            
            h_, mask = cv2.findHomography( bird_eye_coords_,world_coords_,cv2.RANSAC,5.0)
            
            warped = cv2.warpPerspective(masked_edges,h_,(300,600),flags=cv2.INTER_LINEAR)
            
            
            l,r,nxy,lcount,rcount=lane_detector(warped)
            
            final_output=resultant(img,hough_image,lines,lcount,rcount)
            
            
            collage = info_display(img, edges, masked_edges, warped, final_output)
            cv2.imshow('Complete Pipeline', collage)

            # frame_count = 1
            # if frame_count == 1:
            #     # print(os.listdir(directory))
            #     output1 = 'original_image.jpg'
            #     output2 = 'masked_image.jpg'
            #     output3 = 'hough_image.jpg'
            #     output4 = 'edge_image.jpg'
            #     output5 = 'warped_image.jpg'
            #     output6 = 'final_output.jpg'
            #     output7 = 'flipped_output.jpg'
            #     output8 = 'total_outputf.jpg'
                
            #     cv2.imwrite(output1, img)
            #     cv2.imwrite(output2, masked_edges)
            #     cv2.imwrite(output3, hough_image)
            #     cv2.imwrite(output4, edges)
            #     cv2.imwrite(output5, warped)
            #     cv2.imwrite(output6, final_output)
            #     cv2.imwrite(output7, final_output)
            #     cv2.imwrite(output8, collage)
            
            # frame_count += 1
            result.write(final_output)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
cap.release()  
result.release()      
cv2.destroyAllWindows()
