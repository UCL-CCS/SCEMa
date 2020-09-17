# The data passed to py is a list

import numpy as np
import keras as ks

#import tensorflow as tf
def surrogate_model(nlist):

  # Convert inputs list to numpy array
  print(type(nlist[0]))
  inputs = np.array(nlist)
  print(inputs)

  # Some code for surrogate prediction
  # This is a pseudo code
  # model = ks.load('path/name.file')
  # output = model.predict(inputs)

  # convert the data back to a list
  output_list = inputs.tolist()
  print(output_list)
  return output_list
