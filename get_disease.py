import cv2
import numpy as np

from keras.models import load_model
from keras.backend import set_learning_phase
set_learning_phase(0) # Doing this apparently makes the prediction 100% faster

import leaf_segmentation as lf

# These are loaded only once, when we run the program.
path = np.load('nn/v2/categories.npy',allow_pickle=True)
model = load_model('nn/v2/plants.model')
CATEGORIES = []

x = path.flat[0]
for i in range(len(x.keys())):
    CATEGORIES.append(0)

for k,v in x.items():
    CATEGORIES[v] = k    

def prepare_image(im):
    # The image size must be the same as our training pictures (here 128x128)    
    im = cv2.resize(im,(128,128))
    im = np.reshape(im,(-1,128,128,3))
    im = im/255.0
    return im

def get_disease(img,masked): 
    # ============================================================ #
    # Returns a dictionary of diseases that are found on plants for a    
    # recieved image.
    #
    # It uses a tensorflow (implemented with Keras) model to predict 
    # diseases.    
    #
    # The program applies a very soft blur with a kernel size of
    # 2x2 in order to smoothen the image (it has a lot of sharp
    # edges and the threshold functions works better for a smoother
    # image) on which we apply a threshold.
    # 
    # 
    #  >>> recieves as an input two images, img - an original image
    #      (in RGB) and masked - the masked image.
    #  >>> returns a dictionary for each disease found and the 
    #      coordinates around each ocurrences of that disease.
    #
    # ============================================================ #                    

    # Diseases available:
    # Apple Rust, Apple Scab, Bacterial Spot, Black Measles, Black Rot
    # Early blight, Frogeye Spot, Grape Leaf Spot, Healthy, Leaf spot

    # Initializing the disease array to keep track of how many
    # diseases types we have for each image
    diseases=[]
    for i in range(len(CATEGORIES)):
        diseases.append(0)
    # masked = img.copy()
    # masked = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
    # Applying a soft blur on the mask to get rid of those sharp edges.
    masked = cv2.blur(masked, (2,2))

    # I've played with the values and 20, 150 seemed the best,
    # as we want to ignore black (the background), and we also
    # don't want anything too bright.
    # You can change it and try it for yourself.
    _, thresh = cv2.threshold(masked, 20, 150, 0)    
    
    # Find contours on the edges
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
    
    dct={}
    for i in range(len(contours)-1,0,-1):
        if hierarchy[0,i,3] != -1:
            contour = contours[i]
            (x,y,w,h) = cv2.boundingRect(contour)
                    
            # Taking an image only if it is greather than 3x3,
            if w>3 and h>3:                                                      
                # Cutting the rectangle from the original image
                # then normalizing the image
                # and then apply the model on it

                box = img[y:y+h,x:x+w]
                box = prepare_image(box)
                # The .predict() returns an np.array with a percentage
                # for each index of that class. So by using np.argmax(res)
                # we get the index of the class with the most probability
                res = model.predict(box)
                                
                # Check if the detected disease has over 50% confidence and
                # if it has, increase the "diseases[]" + 1 for that disease
                # This works if the model we have has been trained on
                # softmax. If it was trained with a sigmoid activation,
                # then we would need to loop for each disease

                i = np.argmax(res[0])
                
                if not CATEGORIES[i] == "Healthy":                    
                    if res[0][i]>=0.5:             

                        if CATEGORIES[i] in dct:
                            dct[CATEGORIES[i]].append([x,y,w,h])
                        else:
                            dct[CATEGORIES[i]] = [[1],[x,y,w,h]]                                        
                      
    return dct        
        


