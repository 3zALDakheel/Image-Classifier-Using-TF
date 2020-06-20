import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import logging 

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image 
import numpy as np 
import argparse 
import json


parser = argparse.ArgumentParser()

parser.add_argument('--image_path', default='./test_images/cautleya_spicata.jpg', help = 'Please type the image path to get the prediction', type = str)
parser.add_argument('--model' ,default='./best_model.h5' , help = 'Please type the Saved Keras model Path', type = str)
parser.add_argument('--top_k', default=5, help = 'Please type the the number of prediction you want to see', type = int)
parser.add_argument ('--category_names' , default = 'label_map.json', help = 'Path of Categories file name, in JSON format', type = str)
args = parser.parse_args()


image_path =args.image_path
model = args.model
top_k =args.top_k
class_label_map =args.category_names


with open(class_label_map, 'r') as f:
    class_names = json.load(f)
        
        
        
model_load = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})

def process_image(test):
    image_processed = np.squeeze(test)
    image_processed = tf.image.resize(image_processed, (224,224))
    image_processed /=255
    return image_processed


def predict(image_path, model, top_k, class_label_map):
    img = Image.open(image_path)
    transform_image = process_image(np.asanyarray(img))
    prediction = model_load.predict(np.expand_dims(transform_image,axis=0))
    prop, indices = tf.math.top_k(prediction, k=top_k)
    prop = prop.numpy()[0]
    flowername = [class_names[str(value+1)] for value in indices.cpu().numpy()[0]]
    return prop, flowername


if __name__ == '__main__':
    
    #Test 1
    #print(predict('./test_images/hard-leaved_pocket_orchid.jpg', model_load, 3, 'label_map.json'))
    #Output
    #(array([0.974802  , 0.00487249, 0.00365698], dtype=float32), ['hard-leaved pocket orchid', 'bearded iris', 'passion flower'])
    
    
    #test 2
    #print(predict('./test_images/orange_dahlia.jpg', model_load, 2, 'label_map.json'))
    #output
    #(array([0.2548794 , 0.24424188], dtype=float32), ['english marigold', 'orange dahlia'])
   
    #test3
    #python predict.py --image_path ./test_images/cautleya_spicata.jpg --model best_model.h5 --category_names label_map.json
    #output
    #(array([0.6338298 , 0.04464758, 0.0309554 , 0.02454716, 0.01624231],
    # dtype=float32), ['cautleya spicata', 'red ginger', 'wallflower', 'snapdragon', 'yellow iris'])
    print(predict(image_path,model , top_k, class_names))