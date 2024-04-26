import cv2
import numpy as np

from datetime import datetime
time = str(datetime.now())
time2 = "D"+time[:9]+"T"+time[11:13]+"-"+time[14:16]+"-"+time[17:19]
print(time2)

window_name = 'MyWindow'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # You can use different flags as needed

output_path = time2 + '.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))  # Adjust the resolution as needed

while True:
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    cv2.imshow(window_name, img)

    video_writer.write(img)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

