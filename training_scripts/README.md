### Training

Basic flow for training the deployed model:

 - Prepare data using `split_video.py`, splitting a video into frames.
 - Train a model using `supervised_training.py`, assuming labels are available.

There are a few other files here:

 - `labelling.py` is a streamlit app for manually labelling. It works but it's not a good idea.
 - `learn_contrasts.py` is for contrastive pre-training. Also works, but it wasn't better than a conventionally trained model.
 - `make_torchscript_model.py` creates a version of the model that can be loaded in torchscript, if you wanted to use it in a C++ application. Works fine, but it didn't help us with packaging at all.