#!/usr/bin/env python

import sys 
import os
from data import *
from model import *

import datetime

def get_input_data(ID,SLICE,PATH,WLD,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS):
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint16)
    dir_names = glob(PATH+ID+WLD)
    dir_names = natural_sort(dir_names)
    #1st ff, 2nd r2*, 3rd water, 4th fat
    #water ch = 0, ix = 2
    #fat ch = 1, ix = 3
    #ff ch = 2, ix = 0
    #r2* ch = 3, ix = 1
    def zero():
        return 0
    def one():
        return 1
    def two():
        return 2
    def three():
        return 3
    switcher = {
            0: two,
            1: three,
            2: zero,
            3: one
        }
    def ch_to_ix(argument):
        # Get the function from switcher dictionary
        func = switcher.get(argument, "nothing")
        # Execute the function
        return func()
    for ch in range(IMG_CHANNELS):
        dir_name = dir_names[ch_to_ix(ch)]
        _, ch_name = os.path.split(dir_name)
        path = glob(PATH+ID+'/'+ch_name+'/'+ID+'S'+ch_name+'I'+SLICE+'.DCM')
        img = dicom.read_file(path[0])
        img = (img.pixel_array).astype(np.int16)
        img = np.array(img, dtype=np.int16)[np.newaxis,:,:]
        X_test[:,:,:,ch] = img
    
    _, water_ch_name = os.path.split(dir_names[2])
    save_string = ID+'S'+water_ch_name+'I_mask_pred_'+SLICE+'.DCM'
    print('Done!')
    
    return X_test, save_string

def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result*256 + int(b)
    return result

json_file = open('../model/model-3.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('../model/model-3.h5')

model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=[dice_coef,'accuracy'])

width = 256
height = 256
channels = 4
DATA_PATH = '/data/mqir-krug2/ravi/spinevol/'
save_path = '../dcm_data_rc/'

id_suffix = 'E5789'
series_num = '5'
wld = '/'+series_num+'*'
print(id_suffix)
files = glob(DATA_PATH + id_suffix + wld + '/*.DCM')
files = natural_sort(files)
if not os.path.exists(save_path+id_suffix):
    os.makedirs(save_path+id_suffix)
for file in files:
    slice_num = (file.split("I",1)[1]).split(".DCM",1)[0]
    (X_test, save_string) = get_input_data(id_suffix, slice_num, DATA_PATH, wld, width, height, channels)
    # Predict on each slice and save masks as DICOM using the same slice's mask's metadata
    preds_test = model.predict(X_test, verbose=1)
    #print(preds_test.shape)
    preds_test = preds_test.squeeze()
    # Threshold predictions
    preds_test_t = (preds_test > 0.5).astype(np.uint16)
    #print(preds_test_t.shape)
    #plt.imshow(preds_test_t, cmap='gray')
    #plt.show()
    ds = dicom.dcmread(file)
    #print(ds)
    """code_lines = code_file(file)
    exec(code_lines)
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True"""
    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr
    # Change pixel data to predicted values
    ds.PixelData = preds_test_t.tobytes()
    print("Writing test file", save_string)
    filename = save_path + id_suffix + '/' + save_string
    ds.save_as(filename)
    print("File saved.")
    """print('Load file {} ...'.format(filename))
    ds = dicom.dcmread(filename)
    print(ds)"""
