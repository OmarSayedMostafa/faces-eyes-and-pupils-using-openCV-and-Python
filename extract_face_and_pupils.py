from Utilities import *




img = cv2.imread('sample-test/4.jpg')
extract_face_eye_pupil(img)
cv2.imwrite('sample output/sample-4.jpg',img);

img = cv2.imread('sample-test/5.jpg')
extract_face_eye_pupil(img)
cv2.imwrite('sample output/sample-5.jpg',img);


img = cv2.imread('sample-test/7.jpg')
extract_face_eye_pupil(img)
cv2.imwrite('sample output/sample-7.jpg',img);

#live_cam()
#special_function_to_get_the_multiply_of_10_frame_numbers('11.mp4')
saving_video_with_facial_detection('sample-test/11.mp4')