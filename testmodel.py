import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 모델 로드
model = tf.keras.models.load_model(r'C:\Users\kobin\OneDrive\바탕 화면\4학년 1학기\AI프레임워크\예측유지보수\open\train model')

# 테스트 데이터 로드 
test_data = np.load(r'C:\Users\kobin\OneDrive\바탕 화면\4학년 1학기\AI프레임워크\예측유지보수\open\preprocessed_data\test_audios.npy')
test_labels = np.load(r'C:\Users\kobin\OneDrive\바탕 화면\4학년 1학기\AI프레임워크\예측유지보수\open\preprocessed_data\test_labels.npy')

# 데이터 전처리 (필요한 경우)
# test_data = preprocess_test_data(test_data)

# 예측 수행
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# 레이블 데이터 타입 확인
print("Test labels type:", type(test_labels[0]))
print("Predicted classes type:", type(predicted_classes[0]))

# 레이블 인코딩 조정 
if isinstance(test_labels[0], str):
    encoder = LabelEncoder()
    test_labels = encoder.fit_transform(test_labels)

# 성능 평가
print(classification_report(test_labels, predicted_classes))
print("Accuracy:", accuracy_score(test_labels, predicted_classes))
