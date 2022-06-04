# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


## 1. Data Preprocessing
# Preprocessing the Training set (to avoid overfitting)
train_datagen = ImageDataGenerator (
    rescale = 1/255, # divides each pixel by 255
    shear_range = 0.2, # from here all paramters are used to reduce overfitting
    zoom_range = 0.2,
    horizontal_flip = True
)

training_set = train_datagen.flow_from_directory(
    r'D:\Learning\Udemy ML\CNN Dataset\training_set',
    target_size = (64, 64), # resizing the image into a consistent shorter size, will still provide great results, (150, 150) took a lot of time
    batch_size = 32, # number of images in a single batch
    class_mode = 'binary' # outcome type
)

# Preprocessing the Test set (all above features are not applied as original image has to be used for testing)
test_datagen = ImageDataGenerator (rescale = 1/255)

test_set = test_datagen.flow_from_directory(
    r'D:\Learning\Udemy ML\CNN Dataset\test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)


## 2. Building the CNN
# Initializing the CNN
cnn = tf.keras.models.Sequential() # initialises CNN as sequence of layers

# Step 1: Convolution
cnn.add(tf.keras.layers.Conv2D(
    filters = 32, # no. of feature detectors, 32 is classic
    kernel_size = 3, # size of feature detector
    activation = 'relu',
    input_shape = [64, 64, 3] # 64 is for image size we gave above, 3 represents RGB, 1 for B&W
))

# Step 2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size = 2,
    strides = 2 # how many pixels to skip
))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

# Step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4: Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

# Adding the output layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


## 3. Training the CNN
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Training the CNN on the Training set and evaluating it on the test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25) # epoch is smaller as this takes more time then ANN
    # specifying the validation data evaluates the prediction then and there


## 4. Making a single prediction
import numpy as np
import keras.utils.image_utils as image

test_image = image.load_img(r'D:\Learning\Udemy ML\CNN Dataset\single_prediction\cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # this adds batch dimension to the single image
    # as the input expects images to be in batches even though a single image as we have mentioned a batch size above
    # batch size is logically placed in the beginning

result = cnn.predict(test_image)
print('Class Defination: ', training_set.class_indices)

if result[0][0] > 0.5: # first box is to excess the batch, 0.5 increases accuracy
    prediction = 'dog'
else :
    prediction = 'cat'

print('Single Prediction: ', prediction)