# bottomwear_recommender
Recommendation model for bottom wears based on input image.

# Dataset
The dataset used for pre trained model is bottom wear resized images.

# Installation
Use pip to install the requirements.

pip install -r requirements.txt

# Usage
To run the recommender model, simply execute python command with the test.py file along with path to the input image.

python test.py --input "PATH TO THE INPUT FILE"

# Results
I used pre-trained classification model on the dataset that consists of 1050 bottom wear images. The network was trained to extract the feature embeddings from the given input.
For generating the recommendations, I used Sklearn Nearest neighbours. This allowed us to find the nearest neighbours for the given input image. The similarity measure used was euclidean distance. The top 10 recommendations are extracted from the database and their images are displayed.

# Note
train.py is used to generate image_features_embedding.pkl and img_files.pkl respectively.
  
