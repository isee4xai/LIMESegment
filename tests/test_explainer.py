#%%

import sys
import numpy as np
import os

# local modules
sys.path.append(os.path.join (os.path.dirname(__file__), '../src'))

from LIMESegment.LIMESegmentExplainer import LIMESegmentExplainer
from LIMESegment.Utils.data import loadUCRDataID
from LIMESegment.Utils.models import *
from LIMESegment.Utils.metrics import *

OUTPUT_FILE =(os.path.join (os.path.dirname(__file__), 'explanation.png'))

#%%

# load the dataset
dataset_name = 'ECG200'
dataset_idx = 0

train_file = os.path.join (os.path.dirname(__file__), "../Data/ECG200_TRAIN")
test_file  = os.path.join (os.path.dirname(__file__), "../Data/ECG200_TEST")

dataset = loadUCRDataID(train_file = train_file,
                        test_file = test_file )

train_x, train_y, test_x, test_y,_  = dataset

#%%
# train a LSTM model

BATCH_SIZE=8
N_EPOCHS=10

model= make_LSTMFCN_model(train_x.shape[1])
train_LSTMFCN_model(model,
                train_x,
                train_y,
                test_x,
                test_y,
                epochs=N_EPOCHS,
                batch_size=BATCH_SIZE)

# %%
# get the explanation

instance = test_x[0:-1][4]  # the instance to explain
explainer = LIMESegmentExplainer()
explanation = explainer.explain (
                         example = instance, 
                         model = model,
                         model_type="proba",
                         window_size=10,
                         cp=3)

print (explanation[0])
# %%

# plot the explanation

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
explainer.plot_explanation (instance, explanation, ax=ax)
ax.set_ylabel (dataset_name)

plt.savefig (OUTPUT_FILE)
print (f"Explanation saved to {OUTPUT_FILE}")
# %%
