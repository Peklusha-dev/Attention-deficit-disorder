from eye_tracker.detector import PupilDetector
from eye_tracker.utils import save_pupil_data
import cv2
import numpy as np
from pathlib import Path


def process_video(video_path, output_csv="pupil_data.csv", show_result=True):
    detector = PupilDetector(head_tracking=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 1
    pupil_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_time = frame_count / fps
        seconds = int(frame_time)
        microseconds = int((frame_time - seconds) * 1_000_000)
        frame_id = f"frame_0_00_{seconds:02d}_{microseconds:06d}.jpg"
        result_frame, data = detector.detect_pupils_in_frame(frame, frame_id)
        pupil_data.append(data)
        frame_count += 1
        if show_result:
            cv2.imshow("Video and Eye Trajectory", np.hstack((result_frame, detector.trajectory_canvas)))
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    save_pupil_data(pupil_data, output_csv)


if __name__ == "__main__":
    process_video('test_data/video_7.MOV', output_csv='output/pupil_data/pupil_data_7.csv')