from ultralytics import YOLO
import cv2


weights ="/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/yolo_experiments/yolov8m_best.pt"
video_path = "/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/VideoMAE/SN10s-min5c/test/Goal/england_epl__2014-2015__2015-02-21-18-00CrystalPalace1-2Arsenal__1_224p_422.52_428.82|0|MainCameraCenter|Goal.mp4"
video_path = '20230913141256.mp4'
model = YOLO(weights)  # Load your YOLO model


cap = cv2.VideoCapture(video_path)

# Process each frame in the video
frame_num = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = total_frames // 8
sampled_frames = []

for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    if i % frame_interval == 0:
        frame_num += 1
        results = model(frame, verbose=False)
        print("\n", f'Frame {frame_num}, person found {len(results[0].boxes.xyxyn)}')
        print("xyxyn Coordinates:")
        # print('Found', len(results[0].boxes.xyxyn), 'boxes')
        # for xyxyn in results[0].boxes.xyxyn:
        #     print(xyxyn)
        # View results
        for r in results:
            # print(r.boxes.xyxyn.tolist())
            # print([f'{x:.2f}' for e in r.boxes.xyxyn.tolist() for x in e])
            for each in r.boxes.xyxyn.tolist():
                print(*[f'{x:.2f}' for x in each], end=', ')
            # to float decimal only
# Release the video capture object
cap.release()
