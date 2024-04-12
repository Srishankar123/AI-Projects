from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=None,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 50  # Set the batch size

training_set = train_datagen.flow_from_directory(
    'D:\\AI Programming 30 day course\\Day 14 course lessons\\Dataset\\train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'D:\\AI Programming 30 day course\\Day 14 course lessons\\Dataset\\val',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

# Calculate steps per epoch and validation steps
total_train_samples = len(training_set.filenames)
total_test_samples = len(test_set.filenames)
steps_per_epoch = total_train_samples // batch_size
validation_steps = total_test_samples // batch_size

# Train the model with adjusted steps_per_epoch and early stopping
model.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=150,
    validation_data=test_set,
    validation_steps=validation_steps
)

# Save the model
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model1.h5")
print("Saved model to disk")
