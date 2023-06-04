import os
import random
import shutil

def create_validation_folder(data_folder, validation_folder):
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    for root, dirs, files in os.walk(data_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            validation_subfolder = os.path.join(validation_folder, dir_name)

            images = os.listdir(subfolder_path)
            random_image = random.choice(images)
            image_path = os.path.join(subfolder_path, random_image)

            if not os.path.exists(validation_subfolder):
                os.makedirs(validation_subfolder)


            shutil.move(image_path, validation_subfolder)

create_validation_folder('images', 'validation_folder')


