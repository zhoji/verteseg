from data import *
from model import *

# Parameters
width = 256
height = 256
slices = 20
channels = 4
train_path = '../npy_data/train/'
test_path = '../npy_data/test/'

params = {'IMG_WIDTH' : width,
          'IMG_HEIGHT' : height,
          'NUM_SLICES' : slices,
          'IMG_CHANNELS' : channels,
          'TRAIN_PATH' : train_path,
          'TEST_PATH' : test_path'}

(X_train, X_test, y_train, y_test) = get_data(**params)

if os.path.exists('../model/model-3.h5'):
    weights = '../model/model-3.h5'
    model = unet(width, height, channels, weights)
else:
    model = unet(width, height, channels)

# Fit model
model.compile(optimizer=Adam(lr = 1e-4), loss=jaccard_distance_loss, metrics=[dice_coef,'accuracy'])

es = EarlyStopping(patience=50, verbose=1)
mc = ModelCheckpoint('../model/model-3.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_split=0.2, batch_size=4, epochs=500, 
                    callbacks=[es, mc])

# Save model
model_json = model.to_json()
with open("../model/model-3.json", "w") as json_file:
    json_file.write(model_json)

# Evaluate model
json_file = open('../model/model-3.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('../model/model-3.h5')

model.compile(optimizer=Adam(lr = 1e-4), loss=jaccard_distance_loss, metrics=[dice_coef,'accuracy'])
# Evaluate on validation set
model.evaluate(X_test, y_test, verbose=1)

"""
# Predict on train and test
preds_train = model.predict(X_train, verbose=1)
preds_test = model.predict(X_test, verbose=1)
# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Save predictions
np.save("../npy_data/train/preds/preds_train_"+str(channels)+"ch.npy", preds_train)
np.save("../npy_data/train/preds/preds_train_t_"+str(channels)+"ch.npy", preds_train_t)
np.save("../npy_data/test/preds/preds_test_"+str(channels)+"ch.npy", preds_test)
np.save("../npy_data/test/preds/preds_test_t_"+str(channels)+"ch.npy", preds_test_t)
"""

test_ids = next(os.walk(test_path+'imgs/0/'))[2]
for id_ in test_ids:
    (X_test, y_test) = get_testing_data(id_, width, height, slices, channels)
    id_suffix = (id_.split("_",1)[1]).split(".",1)[0]
    # Predict on each testing example and save masks as DICOM
    preds_test = model.predict(X_test, verbose=1)
    # Threshold predictions
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    np.save(test_path+"preds/preds_"+id_suffix, preds_test)
    np.save(test_path+"preds/preds_t_"+id_suffix, preds_test_t)
