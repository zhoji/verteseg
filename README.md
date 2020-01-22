# Automatic Vertebral Segmentation

Get DICOM masks of vertebrae from sagittal IDEAL images

## Setup

Set up a conda environment if you do not have all the packages/compatible versions.
The list of dependencies is listed in `environment.yml`. Set-up environment using conda:
`conda env create -f environment.yml`

The default name of the environment is `vseg`, activate the environment with `source activate vseg`, and deactivate with `source deactivate`.

## Usage

Make sure you are in the code repository.

```
source activate vseg
python get_masks.py [-h] --data_path DATA_PATH --exam EXAM --series SERIES --save_path SAVE_PATH
```
