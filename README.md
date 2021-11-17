# sufficiency-assessment-generation
This project contains the code of the paper "Assessing the Sufficiency of Arguments through Conclusion Generation".

## Installation
To use the code install the modified version of the huggingface transformer repository:

```bash
cd src/repo/transformers
pip install .
```

## ceph_data
All the data used in this work split into input, intermediate and output.
Input: thrid party datasets
Intermediate: pre-processed data
Output: output files

## docs
The annotation study setup and results.
To analyze the results use the jupyter notebooks in /src/ipynb.

## src
The code used to obtain the projects of the work is stored in /src and split based filetypes.

* /src/py/preprocess_ukp.py creates the main dataset used in this work
* pre-processing of the data for training is located at /src/ipynb/data_creation
* /src/sh contains shell files that train the models useing the transformers repository


