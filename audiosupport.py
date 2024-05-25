import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 음원 파일이 있는 디렉토리 경로
data_dir = 'C:/Users/kobin/OneDrive/바탕 화면/4학년 1학기/AI프레임워크/예측유지보수/open'

# 디렉토리 내의 모든 .wav 파일 경로 수집
audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

# 각 오디오 파일에 대한 멜 스펙트로그램 계산 및 시각화
for audio_file in audio_files:
    # 음원 파일 로드
    y, sr = librosa.load(audio_file)

    # 스펙트로그램 계산
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # 스펙트로그램을 디비 스케일로 변환
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # 스펙트로그램 시각화
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()

