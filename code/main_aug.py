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

batch_size = 4
seed = 12398

(X, X_test, y, y_test) = get_data(**params)
X_train, X_val, y_train, y_val = train_test_split(X, y,
    test_size = 0.2,
    random_state = seed)

"""
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    subset='training') # set as training data

validation_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    subset='validation') # set as validation data
"""

image_generator = image_datagen.flow(
    X_train,
    seed=seed,
    batch_size=batch_size)

mask_generator = mask_datagen.flow(
    y_train,
    seed=seed,
    batch_size=batch_size)

train_generator = zip(image_generator, mask_generator)

if os.path.exists('../model/model-4.h5'):
    weights = '../model/model-4.h5'
    model = unet(width, height, channels, weights)
else:
    model = unet(width, height, channels)

# Fit model
model.compile(optimizer=Adam(lr = 1e-3), loss=jaccard_distance_loss, metrics=[dice_coef,'accuracy'])

es = EarlyStopping(patience=50, verbose=1)
mc = ModelCheckpoint('../model/model-4.h5', verbose=1, save_best_only=True)
results = model.fit_generator(train_generator,
                              verbose=1,
                              steps_per_epoch = (len(X_train) // batch_size),
                              validation_data = (X_val, y_val),
                              epochs=500,
                              callbacks=[es, mc])

# Save model
model_json = model.to_json()
with open("../model/model-4.json", "w") as json_file:
    json_file.write(model_json)

# Evaluate model
json_file = open('../model/model-4.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('../model/model-4.h5')

model.compile(optimizer=Adam(lr = 1e-3), loss=jaccard_distance_loss, metrics=[dice_coef,'accuracy'])
# Evaluate on validation set
"""
test_generator = test_datagen.flow(
    X_test, y_test)
model.evaluate_generator(test_generator, verbose=1)
"""
model.evaluate(X_test, y_test, verbose=1)

# Predict on train and test
"""
preds_train = model.predict_generator(train_generator, verbose=1)
preds_test = model.predict_generator(test_generator, verbose=1)
"""
preds_train = model.predict(X_train, verbose=1)
preds_test = model.predict(X_test, verbose=1)
# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Save predictions
np.save("../npy_data/train/preds/preds_train_"+str(channels)+"ch_aug.npy", preds_train)
np.save("../npy_data/train/preds/preds_train_t_"+str(channels)+"ch_aug.npy", preds_train_t)
np.save("../npy_data/test/preds/preds_test_"+str(channels)+"ch_aug.npy", preds_test)
np.save("../npy_data/test/preds/preds_test_t_"+str(channels)+"ch_aug.npy", preds_test_t)
