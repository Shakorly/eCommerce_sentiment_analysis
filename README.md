
# E-commerce Text Classification Project

This project focuses on classifying product descriptions from an E-commerce dataset into four categories: **Electronics**, **Household**, **Books**, and **Clothing & Accessories**. The goal is to build a robust text classification model using Natural Language Processing (NLP) techniques.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Preprocessing](#preprocessing)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

---

## **Project Overview**
The dataset contains product descriptions from an E-commerce website, categorized into four classes. The task is to preprocess the text data, extract features, and train a machine learning model to classify the product descriptions into the correct category.

---

## **Dataset**
The dataset is in `.csv` format with two columns:
- **Category**: The class label (Electronics, Household, Books, Clothing & Accessories).
- **Description**: The product description.

### Dataset Characteristics:
- **Number of Instances**: 50,425
- **Number of Classes**: 4
- **Missing Values**: None

---

## **Installation**
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecommerce-text-classification.git
   cd ecommerce-text-classification

import pandas as pd
data = pd.read_csv("ecommerce_dataset.csv")

from preprocessing import clean_text
data['clean_description'] = data['Description'].apply(clean_text)

from model import train_model
model = train_model(data)


Preprocessing
The text data is preprocessed using the following steps:

Lowercasing: Convert all text to lowercase.

Removing Punctuation: Remove special characters and symbols.

Tokenization: Split text into individual words.

Stopword Removal: Remove common words like "and," "the," etc.

Lemmatization: Reduce words to their base forms.
