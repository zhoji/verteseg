from data import *
from model import *

# Parameters
width = 256
height = 256
slices = 20
channels = 4

params = {'IMG_WIDTH' : width,
          'IMG_HEIGHT' : height,
          'NUM_SLICES' : slices,
          'IMG_CHANNELS' : channels,
          'TRAIN_PATH' : '../npy_data/train/',
          'TEST_PATH' : '../npy_data/test/'}

(X_train, X_test, y_train, y_test) = get_data(**params)

if os.path.exists('../model/model-3.h5'):
    weights = '../model/model-3.h5'
    model = unet(width, height, channels, weights)
else:
    model = unet(width, height, channels)

json_file = open('model-3.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model-3.h5')

model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=[dice_coef,'accuracy'])

# Evaluate on validation set
model.evaluate(X_test, y_test, verbose=1)


