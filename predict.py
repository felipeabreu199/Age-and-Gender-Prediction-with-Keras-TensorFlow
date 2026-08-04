import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img

# load the trained model
model = load_model("models/age_gender_model.h5", compile=False)

# gender dictionary
gender_dict = {0: 'Male', 1: 'Female'}

# path of the dataset
DATASET_PATH = "data-set/datas/UTKFace"

# get list of images
image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')]

# choose an image index to test (change this number)
image_index = 856
filename = image_files[image_index]

# extract real labels from filename
labels = filename.split('_')
real_age = int(labels[0])
real_gender = int(labels[1])

print("File:", filename)
print("Original Gender:", gender_dict[real_gender])
print("Original Age:", real_age)

# load and preprocess the image
img_path = os.path.join(DATASET_PATH, filename)
img = load_img(img_path, color_mode='grayscale')
img = img.resize((128, 128), Image.LANCZOS)
img_array = np.array(img)
img_array = img_array.reshape(1, 128, 128, 1)
img_array = img_array / 255.0

# make prediction
pred = model.predict(img_array)

pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])

print("Predicted Gender:", pred_gender)
print("Predicted Age:", pred_age)

# show image
plt.axis('off')
plt.imshow(img_array.reshape(128, 128), cmap='gray')
plt.title(f"Real: {gender_dict[real_gender]}, {real_age} years\nPred: {pred_gender}, {pred_age} years")
plt.show()