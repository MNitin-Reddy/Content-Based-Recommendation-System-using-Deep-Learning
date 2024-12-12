# Content-Based Recommendation System Using Deep Learning

Using a neural network, this project implements a content-based filtering recommender system for movies. The goal is to predict movie ratings based on user preferences and movie features, leveraging deep learning techniques.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Predictions](#predictions)
    - [Predictions for New Users](#predictions-for-new-users)
    - [Predictions for Existing Users](#predictions-for-existing-users)
    - [Finding Similar Items](#finding-similar-items)
7. [Utilities](#utilities)
8. [Acknowledgments](#acknowledgments)

---

## Overview

This project uses deep learning to implement a content-based recommendation system. The neural network generates user and movie vectors based on input features, and the dot product of these vectors predicts user ratings for movies. The model uses Python and TensorFlow, with additional utilities for data preprocessing and evaluation.

---

## Dataset

The dataset is derived from the [MovieLens ml-latest-small dataset](https://doi.org/10.1145/2827872), focusing on movies released since 2000 in popular genres. Key details include:
- **Users:** 600
- **Movies:** 9,000 (filtered to focus on recent movies)
- **Ratings:** 25,521 (scaled from 0.5 to 5.0)

Each movie is associated with features such as:
- Title
- Release year
- Genres (one-hot encoded for 14 genres)

Users are described by engineered features, such as:
- Per-genre average ratings
- Total rating count
- Average rating

---

## Features

### Movie Features:
- One-hot encoded genres
- Year of release
- Average rating (engineered feature)

### User Features:
- Per-genre average ratings
- User ID (excluded during training)
- Total rating count (excluded during training)
- Average rating (excluded during training)

---

## Model Architecture

The neural network consists of two identical sub-networks for users and items (movies). Each sub-network:
- **Input Layer:** Takes user or movie features
- **Hidden Layers:**
  - Dense (256 units, ReLU activation)
  - Dense (128 units, ReLU activation)
- **Output Layer:** Dense (32 units, linear activation)

The outputs of the user and movie networks are normalized and combined using a dot product to predict ratings.

---

## Training and Evaluation

### Preprocessing:
- Movie and user features are scaled using `StandardScaler`.
- Target ratings are scaled between -1 and 1 using `MinMaxScaler`.

### Training:
- Training data consists of user-movie rating pairs.
- Data is split into training (80%) and testing (20%) sets using `train_test_split`.
- Loss function: Mean Squared Error (MSE)

### Evaluation:
- Achieved a test loss of ~0.081 using the evaluation set.

---

## Predictions

### Predictions for New Users
1. Create a new user vector with desired preferences (e.g., favourite genres).
2. Scale the user vector and replicate it to match the number of movies in the dataset.
3. Predict ratings for all movies.
4. Display top recommended movies based on predicted ratings.

### Predictions for Existing Users
1. Use an existing user’s feature vector.
2. Predict ratings for movies not yet rated by the user.
3. Display recommendations.

### Finding Similar Items
1. Compute feature vectors for movies.
2. Use cosine similarity or other metrics to find movies similar to a target movie.
3. Display similar movie recommendations.

---

## Utilities

### `utils.py`
The `utils.py` script provides helper functions for data loading, preprocessing, and evaluation. Key functions include:

- **`load_data()`**:
  - Loads pre-prepared datasets, including movie and user data, features, and precomputed vectors.
  - Parses movie metadata such as titles and genres.

- **`pprint_train(x_train, features, vs, u_s, maxcount=5, user=True)`**:
  - Nicely prints user or item training data with formatted tables for better visualization.

- **`print_pred_movies(y_p, item, movie_dict, maxcount=10)`**:
  - Displays the top predicted movies for a user along with metadata such as title and genres.

- **`gen_user_vecs(user_vec, num_items)`**:
  - Generates a prediction matrix for user vectors to match the size of movie vectors.

- **`predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler)`**:
  - Predicts user ratings for all movies, returning sorted predictions along with corresponding vectors.

- **`get_user_vecs(user_id, user_train, item_vecs, user_to_genre)`**:
  - Retrieves a user’s vector and corresponding ratings for all movies.

- **`get_item_genres(item_gvec, genre_features)`**:
  - Extracts genres associated with a movie based on its genre vector.

- **`print_existing_user(y_p, y, user, items, ivs, uvs, movie_dict, maxcount=10)`**:
  - Displays predictions for a user already present in the database, including details like genre ratings.

---

### Prerequisites
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Tabulate


## Acknowledgments

- **Dataset:** [MovieLens](https://doi.org/10.1145/2827872)
- **Libraries:** TensorFlow, NumPy, Pandas, Scikit-learn

