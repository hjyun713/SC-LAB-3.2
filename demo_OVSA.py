import argparse
import logging
import time
from utils.log import logger
from utils.parse_config import parse_model_cfg
from utils.timer import Timer
from utils.utils import *
from track import JDETracker
logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
import collections
from mtts.predict_vitals import predict_vitals
import warnings
from mtts.model import MTTS_CAN


# jdetracker 설정단
class mot:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='demo.py')
        self.parser.add_argument('--cfg', type=str, default='cfg/cfg_yolo3.cfg', help='cfg file path')
        self.parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
        self.parser.add_argument('--iou-thres', type=float, default=0.5,
                                 help='iou threshold required to qualify as detected')
        self.parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        self.parser.add_argument('--nms-thres', type=float, default=0.4,
                                 help='iou threshold for non-maximum suppression')
        self.parser.add_argument('--min-box-area', type=float, default=50, help='filter out tiny boxes')
        self.parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
        self.opt = self.parser.parse_args()
        self.cfg_dict = parse_model_cfg(self.opt.cfg)
        self.opt.img_size = [int(self.cfg_dict[0]['width']), int(self.cfg_dict[0]['height'])]
        self.jde = JDETracker(self.opt, frame_rate=30)
        self.frame_id = 0

        setting_str('JDE Tracker Setting...')

    def mot_main(self, frame):
        # Initialize the webcam

        tracker = self.jde  # You may need to adjust the frame_rate
        timer = Timer()

        # Preprocess the frame
        blob, img0 = self.preprocess_frame(frame, self.opt.img_size)

        # Run tracking
        timer.tic()
        blob = torch.from_numpy(blob).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        timer.toc()
        self.frame_id += 1
        # Visualize the result
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        # online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=self.frame_id,
        #                               fps=1. / timer.average_time)
        return [online_ids, online_tlwhs]

    def preprocess_frame(self, frame, img_size=(576,320)):

        # Resize image
        vh, vw = frame.shape[:2]
        w, h = self.get_size(vw, vh, img_size[0], img_size[1])
        img0 = cv2.resize(frame, (w, h))

        # Padded resize
        img, ratio, dw, dh = self.letterbox(img0, height=img_size[1], width=img_size[0])

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img, img0

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def letterbox(self, img, height=320, width=576,
                  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh



def main():
    time_t = 0
    time_save = 0
    cnt = 0
    fps = 0
    frame_rate = 10
    mtts_img_col = 36
    mtts_img_row = 36
    p = r = 0
    s= '-'
    red_color = "\033[31m"
    blue_color = "\033[34m"
    text_color =  "\033[0m"

    # jde, mtts model load
    mot_main = mot()
    setting_str('MTTS_CAN Setting...')
    model = MTTS_CAN(frame_rate, 32, 64, (mtts_img_col,mtts_img_row, 3))
    model.load_weights('mtts\\mtts_can.hdf5')

    # 카메라 디바이스 번호 설정 (일반적으로 0번부터 시작하며, 여러 카메라가 연결되어 있는 경우 변경해야 할 수 있습니다.)
    camera_device_index = '0'

    # 카메라 디바이스를 엽니다.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 카메라의 캡처 해상도를 설정합니다. 4K의 해상도는 (3840, 2160)입니다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    out_frame = collections.deque()
    frame_dict = collections.defaultdict(collections.deque)

    result_dict = collections.defaultdict(list)

    setting_str('Measurement Start...')

    while True:
        # 카메라로부터 프레임을 읽습니다.
        ret, frame = cap.read()
        # print(frame.shape)
        if not ret:
            print("영상을 읽을 수 없습니다.")
            break
        time_t = time.time() # FPS 측정 시작




        for idx, val in frame_dict.items():
            if len(frame_dict[idx]) == 300:
                p,r,s = predict_vitals(frame_dict[idx], model)
                result_dict[idx].append([p,r,s])
                print('=======================================================================================================================')
                print(f"{red_color}ID{idx}{text_color} Pulse(BPM)/Respiration(BPM)/ Stress Value Prediction ( 0 ~ 10 low, 10 ~ 15 normal, 15 ~ High ) : {red_color} {p} / {r} / {s} {text_color}")
                print('=======================================================================================================================')
                for _ in range(30) : frame_dict[idx].popleft()

        SSSI = ''
        if s=='-':
            SSSI='-'
        elif s < 10:
            SSSI = 'L'
        elif s>15:
            SSSI='H'

        else:
            SSSI='M'
        frame = cv2.resize(frame, (576, 320))



        # id와 box_loc을 mot_main에서 return 값으로 받음
        id,box_loc = mot_main.mot_main(frame)

        for id_num, real_id in enumerate(id):

            box_loc[id_num][0] = max(0, box_loc[id_num][0])
            box_loc[id_num][1] = max(0, box_loc[id_num][1])
            box_loc[id_num][2] = max(0, box_loc[id_num][2])
            box_loc[id_num][3] = max(0, box_loc[id_num][3])


            # 얼굴 detection 값을 200x200으로 resize
            fr = frame[(int(box_loc[id_num][1])):(int(box_loc[id_num][1] + box_loc[id_num][3])),(int(box_loc[id_num][0])):(int(box_loc[id_num][0] + box_loc[id_num][2]))]
            re_fr = cv2.resize(fr, (200, 200))
            frame_dict[real_id].append(re_fr)

            frame = cv2.rectangle(frame,(int(box_loc[id_num][0]),int(box_loc[id_num][1])),(int(box_loc[id_num][0] + box_loc[id_num][2]),int(box_loc[id_num][1] + box_loc[id_num][3])),(0,0,255),3) # BBOX DRAW
            frame = cv2.putText(frame,f"ID{real_id}",(int(box_loc[id_num][0]),int(box_loc[id_num][1])-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255)) # ID 표시
            if result_dict[real_id]:
                si = result_dict[real_id][-1][2]
                if si < 10:
                    si = 'L'
                elif si > 15:
                    si = 'H'
                else:
                    si = 'M'

                frame = cv2.putText(frame, f"Pulse : {result_dict[real_id][-1][0]}", (int(box_loc[id_num][0]), int(box_loc[id_num][1]) - 55),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))  # P
                frame = cv2.putText(frame, f"Breath : {result_dict[real_id][-1][1]}",(int(box_loc[id_num][0]), int(box_loc[id_num][1]) - 43), cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 0, 0))  # R
                frame = cv2.putText(frame, f"Stress : {si}",(int(box_loc[id_num][0]), int(box_loc[id_num][1]) - 30), cv2.FONT_HERSHEY_COMPLEX,0.5, (255,0,0))  # S


        time_save = time_save + (time.time() - time_t)
        cnt += 1

        frame = cv2.putText(frame, f"FPS : {fps}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255)) # FPS 표시

        if time_save >= 1 :
            fps = cnt
            time_save = 0
            cnt = 0

        frame = cv2.resize(frame, (1280, 720))
        # 그린 결과 표시
        cv2.imshow("Camera", frame)

        # 'q' 키를 누르면 루프를 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 작업이 끝난 후, 리소스를 해제합니다.
    cap.release()
    cv2.destroyAllWindows()

def setting_str(message):
    width = 70  # 출력할 전체 너비
    padding = (width - len(message)) // 2
    top_bottom_line = '=' * width
    centered_message = f'{"=" * padding}{message}{"=" * padding}'

    print(top_bottom_line)
    print(centered_message)
    print(top_bottom_line)
    print(); print()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
