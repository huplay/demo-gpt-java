# GPT-2 XL parameters

You can download the parameter files from these repos: 
- https://github.com/huplay/GPT2-XL
- https://github.com/huplay/GPT2-XL-part2

Original source: https://github.com/openai/gpt-2
Parameter files can be downloaded using download_model.py,
which points to: https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/<fileName>

The `model.ckpt.data-00000-of-00001` contains the parameters, but it is stored using a TensorFlow internal serialized format.
Instead of implementing the extraction of this file, I extracted the values from the running app. 
