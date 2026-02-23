import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
# Smart-Sorting-Transfer-Learning-for-Identifying-Rotten-Fruits-and-Vegetables
# Smart Sorting using Transfer Learning is an AI-based system that identifies rotten and fresh fruits and vegetables through image classification. It uses pre-trained deep learning models to detect spoilage accurately, enabling automated quality control, reducing food waste, and improving efficiency in markets, farms, and supply chains.
