import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_mel_spectrogram(audio_file):
    try:
        # 오디오 파일 로드
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

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    # 데이터가 있는 디렉토리 경로 설정 (수정 필요)
    data_dir = 'C:\\Users\\kobin\\OneDrive\\바탕 화면\\4학년 1학기\\AI프레임워크\\예측유지보수\\open\\train'

    try:
        # 데이터 디렉토리 안의 모든 오디오 파일에 대해 스펙트로그램 플롯
        for filename in os.listdir(data_dir):
            if filename.endswith(".wav"):
                audio_file = os.path.join(data_dir, filename)
                plot_mel_spectrogram(audio_file)
    except Exception as e:
        print("An error occurred:", e)
