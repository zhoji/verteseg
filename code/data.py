import numpy as np
import pydicom as dicom
import os
from glob import glob
import scipy.ndimage
import re
import sys
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import random
from sklearn.model_selection import train_test_split
from PIL import Image

def load_tif_scan(path):
    slices = []
    files = glob(path + '/*.tif')
    files = natural_sort(files)
    for file in files:
        im = Image.open(file)
        # Convert to Numpy Array
        imarray = np.array(im)
        # Normalize
        #x = (x - 128.0) / 128.0
        x = np.squeeze(imarray)
        slices.append(x)
    slices = np.array(slices)
    #slices = np.flip(slices, 0) #masks were saved in reverse order
    return slices

def get_aaron_data(TEST_ID,IMG_WIDTH,IMG_HEIGHT,NUM_SLICES,IMG_CHANNELS):
    TEST_PATH = '../npy_data/aaron/'
    # Get and resize test images
    #print('Getting test images and masks ... ')
    X_test = np.zeros((NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint16)
    y_test = np.zeros((NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for ch in range(IMG_CHANNELS):
        i = 0
        path = TEST_PATH + 'imgs/' + str(ch) + '/' + TEST_ID
        img = np.load(path)[:,:,:]
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        maskpath = TEST_PATH + 'labels/' + TEST_ID
        mask_ = np.load(maskpath)[:,:,:,np.newaxis]
        mask = np.maximum(mask, mask_)
        for i in range(NUM_SLICES):
            X_test[i,:,:,ch] = img[i]
            y_test[i] = mask[i]
            i+=1
    print('Done!')
    
    return (X_test, y_test)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_scan(path):
    #slices = [dicom.read_file((path + '/' + s) for s in os.listdir(path))]
    slices = []
    for file in glob(path + '/*.DCM'):
        slices.append(dicom.read_file(file))
    slices.sort(key = lambda x: int(x.InstanceNumber)) # sort by slice number
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def show_dcm_info(dataset, path):
    print("Filename.........:", path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()
    
    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    if 'BodyPartExamined' in dataset:
        print("Body Part Examined..:", dataset.BodyPartExamined)
    if 'ViewPosition' in dataset:
        print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

def get_pixels(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=4, cols=5, start_with=0, show_every=1):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12],dpi=300)
    ind = start_with
    for i in range(rows):
        for j in range(cols):
            ax[i,j].set_title('slice %d' % (ind+1))
            ax[i,j].imshow(stack[ind],cmap='gray')
            ax[i,j].axis('off')
            ind = ind + show_every
    plt.show()

def get_data(TRAIN_PATH,TEST_PATH,IMG_WIDTH,IMG_HEIGHT,NUM_SLICES,IMG_CHANNELS):
    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH+'imgs/0/'))[2]
    test_ids = next(os.walk(TEST_PATH+'imgs/0/'))[2]
    
    # Get and resize train images and masks
    #print('Getting train images and masks ... ')
    X_train = np.zeros((len(train_ids)*NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint16)
    y_train = np.zeros((len(train_ids)*NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        for ch in range(IMG_CHANNELS):
            i = 0
            path = TRAIN_PATH + 'imgs/' + str(ch) + '/' + id_
            img = np.load(path)[:,:,:]
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            maskpath = TRAIN_PATH + 'labels/' + id_
            mask_ = np.load(maskpath)[:,:,:,np.newaxis]
            mask = np.maximum(mask, mask_)
            for i in range(NUM_SLICES):
                X_train[n*NUM_SLICES + i,:,:,ch] = img[i]
                y_train[n*NUM_SLICES + i] = mask[i]
                i+=1
    
    # Get and resize test images
    #print('Getting test images and masks ... ')
    X_test = np.zeros((len(test_ids)*NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint16)
    y_test = np.zeros((len(test_ids)*NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    sizes_test = []
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        for ch in range(IMG_CHANNELS):
            i = 0
            path = TEST_PATH + 'imgs/' + str(ch) + '/' + id_
            img = np.load(path)[:,:,:]
            sizes_test.append([img.shape[0], img.shape[1]])
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            maskpath = TEST_PATH + 'labels/' + id_
            mask_ = np.load(maskpath)[:,:,:,np.newaxis]
            mask = np.maximum(mask, mask_)
            for i in range(NUM_SLICES):
                X_test[n*NUM_SLICES + i,:,:,ch] = img[i]
                y_test[n*NUM_SLICES + i] = mask[i]
                i+=1
    print('Done!')
    
    return (X_train, X_test, y_train, y_test)

def get_testing_data(TEST_ID,IMG_WIDTH,IMG_HEIGHT,NUM_SLICES,IMG_CHANNELS):
    TEST_PATH = '../npy_data/test/'
    # Get and resize test images
    #print('Getting test images and masks ... ')
    X_test = np.zeros((NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint16)
    y_test = np.zeros((NUM_SLICES, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for ch in range(IMG_CHANNELS):
        i = 0
        path = TEST_PATH + 'imgs/' + str(ch) + '/' + TEST_ID
        img = np.load(path)[:,:,:]
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        maskpath = TEST_PATH + 'labels/' + TEST_ID
        mask_ = np.load(maskpath)[:,:,:,np.newaxis]
        mask = np.maximum(mask, mask_)
        for i in range(NUM_SLICES):
            X_test[i,:,:,ch] = img[i]
            y_test[i] = mask[i]
            i+=1
    print('Done!')
    
    return (X_test, y_test)

def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))
    
    has_mask = y[ix].max() > 0
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 10), dpi=300)
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='w', levels=[0.5])
    ax[0].set_title('Full Scan')
    
    ax[1].imshow(y[ix].squeeze(), cmap='gray')
    if has_mask:
        ax[1].contour(y[ix].squeeze(), colors='w', levels=[0.5])
    ax[1].set_title('Ground Truth Vertebrae')
    
    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1, cmap='gray')
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='w', levels=[0.5])
    ax[2].set_title('Predicted Vertebrae')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1, cmap='gray')
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='w', levels=[0.5])
    ax[3].set_title('Predicted Vertebrae binary')
    diceco = dice(y[ix].squeeze(),binary_preds[ix].squeeze())
    ax[3].annotate('Dice: '+str(round(diceco,3)),
                   xy=(0.65,0.05),
                   xycoords='axes fraction',
                   c='w',
                   fontsize='large',
                   fontweight='bold')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2,
    #horizontal_flip=True,
    validation_split=0.4)

test_datagen = ImageDataGenerator(rescale=1./255)

image_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2)

mask_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2)
