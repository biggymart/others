import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, 'saved-model')
DEFAULT_TEST_DIR = os.path.join(SCRIPT_PATH, 'test')
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
num_image = 3

model = load_model(os.path.join(DEFAULT_MODEL_DIR, 'mymodel.h5'))

test_images = np.empty((num_image, IMAGE_WIDTH, IMAGE_HEIGHT, 1))

test_list = os.listdir(os.path.join(SCRIPT_PATH, 'test'))
print(test_list)
for idx, img_path in enumerate(test_list):
    img = image.load_img(os.path.join(DEFAULT_TEST_DIR, img_path), target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), color_mode="grayscale")
    img = image.img_to_array(img).astype('float32')/255.
    test_images[idx,:,:,:] = img

y_predict = model.predict(test_images)
print(y_predict.shape)

y_pred = [np.argmax(y) for y in y_predict]
print(y_pred)

file_handler = open('hangul_number_dict.picl', 'rb')
hangul_number = pickle.load(file_handler)
file_handler.close()

reverse_dict = dict(map(reversed, hangul_number.items()))
for value in y_pred:
    print(reverse_dict[value])