""" lambda function wrapper """

import gzip
import pickle
import numpy as np
#import boto3


MODEL_FILE = 'mnist_model.dat.gz'
with gzip.open(MODEL_FILE, 'rb') as f:
    MODEL = pickle.load(f)



def lambda_handler(event, context=None):
    """
        Validate parameters and call the prediction model
        @event: API Gateway's POST body;
        @context: LambdaContext instance;
    """

    # input validation
    assert event, "AWS Lambda event parameter not provided"

    image = event.get("image")
    assert isinstance(image, list)
    image_array = np.array(list)

    # call predicting function
    prediction = predict_mnist(image)

    '''
    label = event.get("label")
    assert isinstance(label, int)

    client = boto3.client('dynamodb')
    try:
        dynamodb.put_item(TableName=imageClassifications, Item={'image': image, 'label': lablel, 'prediction' : prediction})
    except Exception, e:
        print(e)
    '''

    return prediction



def predict_mnist(pixel_array):
    """
        Predict the number (0-9) from an input image
        @pixel_array:  - numpy array of image pixel values (0-255)
    """
    y_predicted = MODEL.predict(pixel_array)

    return y_predicted