import os
import numpy as np
import librosa
import librosa.util

def load_and_preprocess_audio(audio_path, target_length):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        audio = librosa.util.normalize(audio)  # 정규화 추가
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None
    return audio

# 나머지 함수는 동일하되, 예외 처리 로직 추가


def process_directory(directory_path, target_length):
    """Process all .wav files in the given directory."""
    audios = []
    labels = []
    for audio_name in os.listdir(directory_path):
        if audio_name.lower().endswith('.wav'):
            audio_path = os.path.join(directory_path, audio_name)
            audio = load_and_preprocess_audio(audio_path, target_length)
            audios.append(audio)
            labels.append(audio_name[:-4])  # Use file name without '.wav' as label
    return np.array(audios), np.array(labels)

def main(data_dir, target_length):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    preprocessed_dir = os.path.join(data_dir, 'preprocessed_data')
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)
    
    # Process training data
    train_audios, train_labels = process_directory(train_dir, target_length)
    # Process testing data
    test_audios, test_labels = process_directory(test_dir, target_length)
    
    # Save processed data
    np.save(os.path.join(preprocessed_dir, 'train_audios.npy'), train_audios)
    np.save(os.path.join(preprocessed_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(preprocessed_dir, 'test_audios.npy'), test_audios)
    np.save(os.path.join(preprocessed_dir, 'test_labels.npy'), test_labels)
    
    print("Data has been saved successfully.")
    print("Train audios shape:", train_audios.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test audios shape:", test_audios.shape)
    print("Test labels shape:", test_labels.shape)

if __name__ == "__main__":
    data_dir = r'C:\Users\kobin\OneDrive\바탕 화면\4학년 1학기\AI프레임워크\예측유지보수\open'
    target_length = 22050  # For example, 1 second at 22050Hz
    main(data_dir, target_length)
