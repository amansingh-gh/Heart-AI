import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# 1. Path set karein (Ensure 'ECG_Dataset' folder contains your 4 sub-folders)
data_path = 'ECG_Dataset' 

# 2. Data Loading & Augmentation
# Hum images ko resize karke 224x224 banayenge
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Printing class names to confirm
print("Classes detected:", train_data.class_indices)

# 3. Simple & Powerful CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(4, activation='softmax') # 4 folders = 4 outputs
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Training with Early Stopping (Saves time!)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

print("\n--- Starting Training ---")
model.fit(
    train_data, 
    validation_data=val_data, 
    epochs=15, 
    callbacks=[early_stop]
)

# 5. Save the Model
model.save('ecg_vision_model.h5')
print("\nSuccess! 'ecg_vision_model.h5' has been created in your folder.")