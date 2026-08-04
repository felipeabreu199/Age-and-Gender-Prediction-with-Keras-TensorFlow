# Age and Gender Prediction with Keras & TensorFlow

<img width="1414" height="2000" alt="ada" src="https://github.com/user-attachments/assets/5737c629-77e2-40fb-b676-82b090ba055f" />

<img width="800" height="429" alt="ScreenRecording2026-08-03221309-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/0b2cc2f8-b9ce-4ef5-a997-177e8d51c36a" />


A deep learning project that predicts **gender** and **age** from facial images using a Convolutional Neural Network (CNN).

Built with TensorFlow/Keras and trained on the **UTKFace** dataset.

## Features

- Multi-output CNN (Gender Classification + Age Regression)
- Grayscale image preprocessing (128x128)
- BatchNormalization + Dropout for better generalization
- Training visualization (Accuracy & MAE graphs)
- Model saved as `.h5` for easy inference

---

## Dataset

**UTKFace** - Large-scale face dataset with age, gender and ethnicity labels.

- Age range: 0 to 116 years
- Gender: 0 = Male, 1 = Female
- More than 20,000 aligned and cropped face images

Download: [UTKFace on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

---

## Project Structure
```text
Age-Gender-Prediction/
│
├── .venv/                             # virtual environment for the project dependencies
│
├── data-set/
│   └── datas/                         # dataset directory
│       ├── crop_part1/                # cropped face images
│       ├── UTKFace/                   # original utkface dataset
│       └── utkface_aligned_cropped/   # aligned and cropped face images
│
├── models/
│   └── age_gender_model.h5            # trained model generated after training
│
├── .gitignore                         # files and folders ignored by git
├── predict.py                         # script for making predictions using the trained model
├── train.py                           # script for training the age and gender model
├── requirements.txt                   # list of required python packages
└── README.md                          # project documentation and usage instructions
```

# How to Use

## Training the Model

If you do not want to use the pre-trained `age_gender_model.h5` file provided with this project, you can train your own model by following these steps:

1. Run the `train.py` script.

2. Wait until the training process is completed. By default, the model will run for **30 epochs**, but you can modify this value according to your needs.

3. After training finishes, the trained model will be saved as `age_gender_model.h5`. You can rename this file or change the output name in the code if desired.

4. During training, performance graphs will be generated to help visualize metrics such as accuracy and loss, making it easier to understand how the model is learning.


## Making Predictions

After training the model (or if you are using the provided `age_gender_model.h5` file), you can make predictions using the `predict.py` script.

Before running the script, select the image you want to analyze by changing the following variable:

```python
image_index = ...
```
for example
```python
image_index = 42
```
