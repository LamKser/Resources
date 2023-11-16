import numpy as np
import cv2

# scale for plotting with different image sizes 
SCALE_TEXT = [0.03, 0.1]    # [x, y]
SCALE_KEYPOINTS = 0.05
SCALE_FONT = 0.01
SCALE_THICKNESS = 0.006
KEYPOINTS_THICKNESS = 0.02
KEYPOINTS_COLOR = {
    0: ["top_left", (124,252,0)], # lawngreen
    1: ["top_right", (255,0,0)], # red
    2: ["bot_right", (0,191,255)], # deepskyblue
    3: ["bot_left", (255,20,147)],  # deeppink
}

def draw_4_keypoints_with_labels(image, keypoints):
    img = image.copy()
    h, w = img.shape[:2]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fontScale = min(h, w) / (25 / SCALE_FONT)

    # set up thickness
    text_thickness = int(min(h, w) * SCALE_THICKNESS)
    keypoints_thickness = int(min(h, w) * KEYPOINTS_THICKNESS)

    # TEXT
    text_kp = np.array([
        [keypoints[0][0] + w * SCALE_TEXT[0],       keypoints[0][1] + h * SCALE_TEXT[1]],
        [keypoints[1][0] - 9 * (w * SCALE_TEXT[0]), keypoints[1][1] + h * SCALE_TEXT[1]],
        [keypoints[2][0] - 9 * (w * SCALE_TEXT[0]), keypoints[2][1] - h * SCALE_TEXT[1]],
        [keypoints[3][0] + w * SCALE_TEXT[0],       keypoints[3][1] - h * SCALE_TEXT[1]]
    ]).astype(int)

    # new_kp = np.array([
    #     [keypoints[0][0] + w * SCALE_KEYPOINTS, keypoints[0][1] + h * SCALE_KEYPOINTS],
    #     [keypoints[1][0] - w * SCALE_KEYPOINTS, keypoints[1][1] + h * SCALE_KEYPOINTS],
    #     [keypoints[2][0] - w * SCALE_KEYPOINTS, keypoints[2][1] - h * SCALE_KEYPOINTS],
    #     [keypoints[3][0] + w * SCALE_KEYPOINTS, keypoints[3][1] - h * SCALE_KEYPOINTS]
    # ]).astype(int)

    # draw keypoints
    for i, kp in enumerate(keypoints):
        cv2.circle(img, kp, keypoints_thickness, KEYPOINTS_COLOR[i][1], -1)
        # cv2.circle(img, new_kp[i], keypoints_thickness, KEYPOINTS_COLOR[i][1], -1)
        cv2.putText(img, KEYPOINTS_COLOR[i][0], text_kp[i], 
                    cv2.FONT_HERSHEY_SIMPLEX, 5 * fontScale, KEYPOINTS_COLOR[i][1], text_thickness, cv2.LINE_8)
    return img


def warp_4_points_image(image, points):
    img = image.copy()
    # width of warpped image
    widthA = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    widthB = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))   
    maxWidth = max(int(widthA), int(widthB))
    # height of warpped image
    heightA = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    heightB = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners = [[0, 0], 
                           [maxWidth - 1, 0], 
                           [maxWidth - 1, maxHeight - 1], 
                           [0, maxHeight - 1]]
    
    M = cv2.getPerspectiveTransform(np.float32(points), np.float32(destination_corners))
    warpped_img = cv2.warpPerspective(img, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)

    return warpped_img

