import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LayerNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def load_data():
    data_dir = r'C:\Users\kobin\OneDrive\바탕 화면\4학년 1학기\AI프레임워크\예측유지보수\open\preprocessed_data'
    train_audios = np.load(os.path.join(data_dir, 'train_audios.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))

    # 레이블 인코딩
    labels_index = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    train_labels = np.array([labels_index[label] for label in train_labels])
    train_labels = to_categorical(train_labels)  # 원-핫 인코딩

    return train_audios, train_labels

def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        LayerNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # 데이터 로딩
    audios, labels = load_data()
    audios = np.expand_dims(audios, -1)  # LSTM에 맞게 차원 추가

    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(audios, labels, test_size=0.2, random_state=42)

    # 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes)

    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        'predictive_maintenance_checkpoint.keras',
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    # 모델 학습
    model.fit(
        X_train, y_train,
        epochs=30,  # 에포크 수 변경
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback]  # ReduceLROnPlateau 및 EarlyStopping 콜백 제거
    )

    # 모델 저장
    model.save('predictive_maintenance_final_model.keras')
    print("Model has been trained and saved successfully.")

if __name__ == "__main__":
    main()
