markdown_content = """<div align="center">

# K-Nearest Neighbors (KNN) — Diabetes Classification

**Computational Science Laboratory Activity**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge)
![Built from Scratch](https://img.shields.io/badge/Frameworks-None-success?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

</div>

---

## 👥 Group 6

- **Muhammed Shariff Sumagka**
- **Lara Rain Fuentes**
- **Gerard Carl Palma**

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Dataset](#-dataset)
4. [Requirements](#-requirements)
5. [Usage](#-usage)
6. [Output & Dashboards](#-output--dashboards)
7. [License](#-license)

---

## 📌 Overview

This repository contains a complete, manual implementation of the K-Nearest Neighbors (KNN) machine learning algorithm built entirely from scratch in Python. The objective of this laboratory activity is to predict the onset of diabetes based on diagnostic measurements without relying on high-level machine learning frameworks like `scikit-learn`.

By building the algorithm from the ground up, this project demonstrates a fundamental understanding of distance metrics, data preprocessing pipelines, and algorithmic time complexity.

## ✨ Key Features

- **Strict Data Integrity:** Custom train/test splitting is performed _before_ imputation and scaling to completely eliminate data leakage, ensuring mathematically honest evaluation metrics.
- **Algorithmic Optimization:** Utilizes tuple conversion and Python's built-in `heapq` module to extract nearest neighbors, reducing sorting time complexity from $\mathcal{O}(N \log N)$ to $\mathcal{O}(N \log K)$.
- **Comprehensive Preprocessing:** Implements median imputation for biologically impossible zero-values and Min-Max Normalization to standardize geometric Euclidean distances.
- **Advanced Data Visualization:** Generates a fully responsive, symmetric, dark-mode dashboard using `matplotlib` to visually map accuracy curves, precision/recall metrics, feature distributions, and custom-styled confusion matrices.

## 🗄️ Dataset

The project utilizes the `diabetes-k-nn.csv` dataset, consisting of 768 patient records mapped across 8 physiological features:

1. `Pregnancies`
2. `Glucose`
3. `BloodPressure`
4. `SkinThickness`
5. `Insulin`
6. `BMI`
7. `DiabetesPedigreeFunction`
8. `Age`

_Note: The target variable is `Outcome` (0 = Non-diabetic, 1 = Diabetic)._

## ⚙️ Requirements

This script relies primarily on Python's standard library to handle the core algorithmic logic, requiring only `matplotlib` to render the final aesthetic dashboard.
