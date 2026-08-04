import os
import warnings
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, BatchNormalization

warnings.filterwarnings("ignore")


# 1. load dataset
DATASET_PATH = "data-set/datas/UTKFace"

image_paths = []
age_labels = []
gender_labels = []
ethnicity_labels = []

for filename in tqdm(os.listdir(DATASET_PATH)):
    try:
        labels = filename.split("_")

        # ignore files that do not follow the pattern
        if len(labels) < 4:
            continue

        age = int(labels[0])
        gender = int(labels[1])
        ethnicity = int(labels[2])

        image_path = os.path.join(DATASET_PATH, filename)

        image_paths.append(image_path)
        age_labels.append(age)
        gender_labels.append(gender)
        ethnicity_labels.append(ethnicity)

    except ValueError:
        continue

# create dataframe
df = pd.DataFrame({
    "image": image_paths,
    "age": age_labels,
    "gender": gender_labels,
    "ethnicity": ethnicity_labels
})

print(df.head())
print("total images:", len(df))
print("\ngender distribution:")
print(df["gender"].value_counts())
print("\nethnicity distribution:")
print(df["ethnicity"].value_counts())

# map genders for label
gender_dict = {0: 'Male', 1: 'Female'}


# 2. exploratory data analysis
# display first image
Img = Image.open(df['image'][0])
plt.axis('off')
plt.imshow(Img)
plt.show()

# age distribution
sns.displot(df['age'])
plt.show()

# gender count
sns.countplot(x=df['gender'])
plt.show()

# grid of images
plt.figure(figsize=(20, 20))
files = df.iloc[0:25]

for index, (file, age, gender, ethnicity) in enumerate(files.itertuples(index=False)):
    plt.subplot(5, 5, index + 1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} | {gender_dict[gender]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# 3. feature extraction
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

X = extract_features(df['image'])
print("features shape:", X.shape)

# normalize images
X = X / 255.0

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

input_shape = (128, 128, 1)


# 4. model creation
inputs = Input(shape=input_shape)

# convolutional layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)

# gender branch
gender_dense = Dense(256, activation='relu')(x)
gender_dense = Dropout(0.4)(gender_dense)
gender_out = Dense(1, activation='sigmoid', name='gender_out')(gender_dense)

# age branch
age_dense = Dense(256, activation='relu')(x)
age_dense = Dropout(0.4)(age_dense)
age_out = Dense(1, activation='relu', name='age_out')(age_dense)

model = Model(inputs=inputs, outputs=[gender_out, age_out])

model.compile(
    loss={'gender_out': 'binary_crossentropy', 'age_out': 'mae'},
    optimizer='adam',
    metrics={'gender_out': 'accuracy', 'age_out': 'mae'}
)

model.summary()


# 5. train model
history = model.fit(
    x=X,
    y={'gender_out': y_gender, 'age_out': y_age},
    batch_size=32,
    epochs=30,
    validation_split=0.2
)


# 6. plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['gender_out_accuracy'], label='train accuracy')
plt.plot(history.history['val_gender_out_accuracy'], label='val accuracy')
plt.title('gender accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['age_out_mae'], label='train mae')
plt.plot(history.history['val_age_out_mae'], label='val mae')
plt.title('age mae')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend()

plt.tight_layout()
plt.show()


# 7. save model
model.save("age_gender_model.h5")
print("model saved as age_gender_model.h5")