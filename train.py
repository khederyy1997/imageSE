import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_directory = 'images'
valid_directory = 'validation_folder'
save_model_path = 'path/to/save/model.h5'


input_shape = (224, 224, 3)
num_classes = len(os.listdir(train_directory))

base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
epochs = 10

train_generator = train_datagen.flow_from_directory(train_directory, target_size=input_shape[:2],
                                                    batch_size=batch_size, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_directory, target_size=input_shape[:2],
                                                    batch_size=batch_size, class_mode='categorical')

model.fit(train_generator, steps_per_epoch=train_generator.n // batch_size,
          validation_data=valid_generator, validation_steps=valid_generator.n // batch_size,
          epochs=epochs)

loss, accuracy = model.evaluate(valid_generator)
print(f'Validation Accuracy: {accuracy}')

model.save(save_model_path)
print(f'Model saved to: {save_model_path}')

# unseen_image_path = 'path/to/unseen_image.jpg'

# # Preprocess the unseen image
# unseen_image = load_img(unseen_image_path, target_size=input_shape[:2])
# unseen_image = img_to_array(unseen_image)
# unseen_image = np.expand_dims(unseen_image, axis=0)
# unseen_image = unseen_image / 255.0

# # Extract the embeddings using the trained model
# embeddings = model.predict(unseen_image)
# print(f'Embeddings shape: {embeddings.shape}')
