from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.utils import CustomObjectScope
import FaceLib
def img_encoding(img_path):
    engine = FaceLib.Align_Face_Engine()
    alignedFace = engine.align_face(img_path)
    with CustomObjectScope({'tf': tf}):
        model = load_model('./model/nn4.small2.v1.h5')
    x_train = np.array([alignedFace])
    y = model.predict_on_batch(x_train)
    return y
print img_encoding('img/Peng.png')
