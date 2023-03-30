# GPT-2 LARGE parameters

You can download the parameter files from this repo: https://github.com/huplay/GPT2-LARGE

Original source: https://github.com/openai/gpt-2
Parameter files can be downloaded using download_model.py,
which points to: https://openaipublic.blob.core.windows.net/gpt-2/models/774M/<fileName>

The `model.ckpt.data-00000-of-00001` contains the parameters, but it is stored using a TensorFlow internal serialized format.
Instead of implementing the extraction of this file, I extracted the values from the running app. 
