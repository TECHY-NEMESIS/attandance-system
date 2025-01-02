import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

dataset_path = r'C:\Users\secur\attendance_system_2\dataset1'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

class_labels = train_generator.class_indices
with open('class_labels.json', 'w') as json_file:
    json.dump(class_labels, json_file)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

model.save('face_recognition_model.h5')

with open('class_labels.json', 'w') as json_file:
    json.dump(train_generator.class_indices, json_file)

print("Model has been trained and saved as 'face_recognition_model.h5'.")
#r'C:\Users\secur\attendance_system_2\dataset1'
