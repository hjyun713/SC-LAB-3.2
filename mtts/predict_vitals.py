import scipy.io
import sys
import numpy as np
import scipy.io
sys.path.append('mtts\\')
from scipy.signal import butter, find_peaks
from inference_preprocess import preprocess_raw_video, detrend
import math







#     parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
#     parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
def predict_vitals(frame_lst, model):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    batch_size = 100 # batch size (multiplier of 10)
    frame_rate = fs = 30 # sampling rate of your video


    predict_pulse = 0
    predict_res = 0
    predict_stress = 0

    frame_dict = dict()
    frame_dict['height'] = 200
    frame_dict['width'] = 200
    frame_dict['frame_cnt'] = 300

    dXsub = preprocess_raw_video(frame_lst, frame_dict, dim=36)  # sample_data_path 현재 웹캠
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth) * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]






    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)


    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred_filtered = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred_filtered = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))


    p_peaks, _ = find_peaks(pulse_pred_filtered,height=0., distance=15)
    r_peaks, _ = find_peaks(resp_pred_filtered, distance=60)
    # Plotting the two signals in one graph with different colors
    # plt.plot(pulse_pred_filtered, color='blue', label='Pulse')
    # plt.plot(resp_pred_filtered, color='red', label='Respiration')
    # plt.plot(p_peaks, pulse_pred_filtered[p_peaks], 'ro', markersize=5, label='Pulse Peaks')
    # plt.plot(r_peaks, resp_pred_filtered[r_peaks], 'ko', markersize=6, label='Res Peaks')
    # # x 축 설정
    # plt.xticks(np.arange(0, 301, 5), labels=['' if i % 10 != 0 else str(i * 5) for i in range(0, 61)])


    # print(peaks)


    # # 그리드 설정
    # plt.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5, color='gray')
    # plt.legend(loc='lower left')  # Display the legend
    # plt.show()  # Show the plot


    resp_temp = []
    pulse_temp = []

    # 예상 평균 심박/호흡 구하기
    if len(r_peaks) >= 2:
        for res_index in range(1,len(r_peaks),1):
            resp_temp.append(math.ceil(60 / ((r_peaks[res_index] - r_peaks[res_index - 1]) / frame_rate)))

    if len(p_peaks)>=2:
        for pul_index in range(1,len(p_peaks),1):
            pulse_temp.append(math.ceil(60 / ((p_peaks[pul_index] - p_peaks[pul_index - 1]) / frame_rate)))


    print('Pulse Peak vals : ',pulse_temp)
    print('Respiration Peak vals : ', resp_temp)


    # 예상 맥박,호흡
    predict_pulse = int(np.mean(pulse_temp))
    predict_res = int(np.mean(resp_temp))

    bin_size = 1.5  # 칸을 기준으로 50ms가 1.5사이즈로 해야함
    num_bins = int(np.ceil((max(pulse_temp) - min(pulse_temp)) / bin_size)) # 구간의 갯수 구하기


    bin_counts, bin_edges = np.histogram(pulse_temp, bins=num_bins)
    Amo = max(bin_counts) / num_bins * 100
    Mo = 0.05
    MxDMn = (max(pulse_temp) - min(pulse_temp)) / 30
    SI = int(math.sqrt(Amo / (Mo * MxDMn)))


    predict_stress = SI            # 10보다 낮음: 낮은 스트레스
                                   # 10-15 정상
                                   # 15 이상: 높은 스트레스

    #
    # plt.hist(pulse_temp, bins=num_bins, edgecolor='black')
    #
    # plt.xlabel('Pulse Temp')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Pulse Temp')
    #
    # plt.show()


    return predict_pulse,predict_res,predict_stress