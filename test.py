import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, dest='inputpath', required=True, help='Path to the input file')
args = parser.parse_args()

# image feature vector embeddings as .pkl file
features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

print(np.array(features_list).shape)

# Initialising model utilizing transfer learning from ResNet50
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# img = image.load_img(r'C:\Users\Sudhanshu Loshali\Downloads\BottomRecommender\sample\35469113THD.png',target_size=(224,224))

img = args.inputpath
updated_img = cv2.imread(img)
resized_img = cv2.resize(updated_img, (224,224))

img_array = image.img_to_array(resized_img)
expand_img = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expand_img)
result_to_resnet = model.predict(preprocessed_img)
flatten_result = result_to_resnet.flatten()
# normalizing
result_normlized = flatten_result / norm(flatten_result)

# Finding 10 nearest neighbours using euclidean distance
neighbors = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
neighbors.fit(features_list)

distance, indices = neighbors.kneighbors([result_normlized])

print(indices)

for file in indices[0][1:11]:
    print(img_files_list[file])
    tmp_img = cv2.imread(img_files_list[file])
    tmp_img = cv2.resize(tmp_img, (250, 250))
    cv2.imshow("output", tmp_img)
    cv2.waitKey(0)
