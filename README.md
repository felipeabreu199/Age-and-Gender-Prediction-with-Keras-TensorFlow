# Age and Gender Prediction with Keras & TensorFlow

A deep learning project that predicts **gender** and **age** from facial images using a Convolutional Neural Network (CNN).

Built with TensorFlow/Keras and trained on the **UTKFace** dataset.

---

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

```bash
Age-Gender-Prediction/
├── data-set/
│   └── datas/
│       └── UTKFace/          # dataset images
├── train.py                  # training script
├── age_gender_model.h5       # trained model (generated after training)
├── requirements.txt
└── README.md
