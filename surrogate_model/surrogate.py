# The data passed to py is a list

import numpy as np
from keras import models
from pickle import load

#import tensorflow as tf
def surrogate_model(nlist):
  # Convert inputs list to numpy array
  #print(type(nlist[0]))
  #print(nlist)
  inputs = np.array(nlist)
  inputs_reshape = inputs.reshape(1, -1)

  # Load scaler and transform the X_test
  scaler = load(open('./surrogate_model/scaler.pkl', 'rb'))
  inputs_scaled = scaler.transform(inputs_reshape)
  #print(inputs_scaled)

  # Load model and predict
  model = models.load_model("./surrogate_model/model_small_uniaxial.bin")
  output_pred = model.predict(inputs_scaled)
  #print(output_pred)

  # convert the data back to a list
  output_list = output_pred.tolist()[0]
  #print(output_list)

  return output_list
