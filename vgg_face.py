from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D,MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from mtcnn import MTCNN
import cv2
import json
import requests

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('models/vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

def preprocess_image(image_path):
    img = load_img(image_path,target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    return img






face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        detected_face = img[y:y+h, x:x+w]
        detected_face = cv2.resize(detected_face, (224, 224))
        img_pixels = img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = img_pixels / 255.0
        return img_pixels
    else:
        im = cv2.resize(img, (224, 224))
        img_pixels = img_to_array(im)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = img_pixels / 255.0

        return img_pixels





def find_cosine_similarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation),test_representation)
    b = np.sum(np.multiply(source_representation,source_representation))
    c = np.sum(np.multiply(test_representation,test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))



# def verify_face(img1, img2):
#     img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
#     img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
#     cosine_similarity = find_cosine_similarity(img1_representation,img2_representation)
#     print("Cosine similarity: ",cosine_similarity)



# def find_similar_images(query_image_path, file_directory, top_k=10):
#     query_embedding = vgg_face_descriptor.predict(preprocess_image(query_image_path))[0, :]
#     similar_images = []

#     for root, dirs, files in os.walk(file_directory):
#         for file in files:
#             image_path = os.path.join(root, file)
#             image_embedding = vgg_face_descriptor.predict(preprocess_image(image_path))[0, :]
#             similarity = find_cosine_similarity(query_embedding, image_embedding)
#             similar_images.append((file, similarity))

#     similar_images.sort(key=lambda x: x[1])  
#     top_similar_images = similar_images[:top_k]

#     return top_similar_images



import os
from os.path import basename

def get_image_names(directory):
    image_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):  # Adjust the file extensions as per your requirement
                image_names.append(file)
    return image_names





# def detect_faces_in_directory(directory):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     image_names_with_faces = []

#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith((".jpg", ".jpeg", ".png")):  # Adjust the file extensions as per your requirement
#                 image_path = os.path.join(root, file)
#                 image = cv2.imread(image_path)
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#                 if len(faces) > 0:
#                     image_names_with_faces.append(file)

#     return image_names_with_faces


# im_with_faces= detect_faces_in_directory("gallery")
# print(im_with_faces)


# directory = "gallery"
# image_names_with_faces = detect_faces_in_directory(directory)
# print(image_names_with_faces)


# def find_similar_images(query_image_paths, file_directory, top_k=10):
#     results = {}

#     for query_image_path in query_image_paths:
#         query_name = os.path.splitext(basename(query_image_path))[0]
#         query_embedding = vgg_face_descriptor.predict(preprocess_image(query_image_path))[0, :]
#         similar_images = []

#         for root, dirs, files in os.walk(file_directory):
#             for file in files:
#                 image_path = os.path.join(root, file)
#                 image_embedding = vgg_face_descriptor.predict(preprocess_image(image_path))[0, :]
#                 similarity = find_cosine_similarity(query_embedding, image_embedding)
#                 similar_images.append((file, similarity))

#         similar_images.sort(key=lambda x: x[1])
#         top_similar_images = similar_images[:top_k]
#         file_names = [file for file, _ in top_similar_images]
#         results[query_name] = file_names

#     return results



# directory = "query"
# query_image_paths = get_image_names(directory)

# file_directory = "g"
# similar_images = find_similar_images(query_image_paths, file_directory, top_k=10)

# for query_name, similar_file_names in similar_images.items():
#     print("Query:", query_name)
#     print("Similar Images:", similar_file_names)
#     print()

# query_image_path = "4.jpg"
# file_directory = "gallery"
# similar_images = find_similar_images(query_image_path, file_directory, top_k=10)
# file_names = [file for file, _ in similar_images]
# # for image_name, similarity in similar_images:
# #     print("Image:", image_name, "Similarity:", similarity)
# print(file_names)

# import os
# from os.path import basename

# def find_similar_images(query_images_directory, file_directory, top_k=10):
#     results = {}

#     query_image_paths = [os.path.join(query_images_directory, file) for file in os.listdir(query_images_directory) if file.endswith((".jpg", ".jpeg", ".png"))]

#     for query_image_path in query_image_paths:
#         query_name = os.path.splitext(basename(query_image_path))[0] + ".jpg"  # Update query_name assignment
#         query_embedding = vgg_face_descriptor.predict(preprocess_image(query_image_path))[0, :]
#         similar_images = []

#         for root, dirs, files in os.walk(file_directory):
#             for file in files:
#                 image_path = os.path.join(root, file)
#                 image_embedding = vgg_face_descriptor.predict(preprocess_image(image_path))[0, :]
#                 similarity = find_cosine_similarity(query_embedding, image_embedding)
#                 similar_images.append((file, similarity))

#         similar_images.sort(key=lambda x: x[1])
#         top_similar_images = similar_images[:top_k]
#         file_names = [file for file, _ in top_similar_images]
#         results[query_name] = file_names

#     return results


# query_images_directory = "query"
# file_directory = "gallery"
# similar_images = find_similar_images(query_images_directory, file_directory, top_k=10)


import os
import numpy as np
import pickle

def calculate_embeddings(file_directory):
    embeddings = {}

    for root, dirs, files in os.walk(file_directory):
        for file in files:
            image_path = os.path.join(root, file)
            image_embedding = vgg_face_descriptor.predict(preprocess_image(image_path))[0, :]
            embeddings[file] = image_embedding

    # Save the embeddings to a file
    with open("gallery_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

def find_similar_images(query_images_directory, file_directory, top_k=10):
    results = {}

    # Load gallery embeddings from file
    with open("gallery_embeddings.pkl", "rb") as f:
        gallery_embeddings = pickle.load(f)

    query_image_paths = [os.path.join(query_images_directory, file) for file in os.listdir(query_images_directory) if file.endswith((".jpg", ".jpeg", ".png"))]

    for query_image_path in query_image_paths:
        query_name = os.path.splitext(os.path.basename(query_image_path))[0] + ".jpg"  # Update query_name assignment
        query_embedding = vgg_face_descriptor.predict(preprocess_image(query_image_path))[0, :]
        similar_images = []

        for file, gallery_embedding in gallery_embeddings.items():
            similarity = find_cosine_similarity(query_embedding, gallery_embedding)
            similar_images.append((file, similarity))

        similar_images.sort(key=lambda x: x[1])
        top_similar_images = similar_images[:top_k]
        file_names = [file for file, _ in top_similar_images]
        results[query_name] = file_names

    return results

# Calculate and save the embeddings for the gallery images
gallery_directory = "gallery"
# gallery_embeddings = calculate_embeddings(gallery_directory)

# Find similar images using the pre-calculated embeddings
query_directory = "query"
similar_images = find_similar_images(query_directory, gallery_directory)


# print(similar_images)

# query_image_paths = "q"
# file_directory = "gallery"
# similar_images = find_similar_images(query_image_paths, file_directory, top_k=10)

# print(similar_images)
# for query_image_path, similar_file_names in similar_images.items():
#     print("Query Image:", query_image_path)
#     print("Similar Images:", similar_file_names)
#     print()


mydata = dict()
mydata['groupname'] = "Without_Face" 
mydata["images"] = similar_images




def submit(results, url="https://competition-production.up.railway.app/results/"):
 res = json.dumps(results)
 response = requests.post(url, res)
 try:
    result = json.loads(response.text)
    print(f"accuracy is {result['results']}")
    return result
 except json.JSONDecodeError:
    print(f"ERROR: {response.text}")
 return None

submit(mydata)