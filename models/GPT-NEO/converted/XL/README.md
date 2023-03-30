# GPT-NEO-XL parameters

(Same as EleutherAI GPT-NEO-1.3B)

You can download the parameter files from this repo: https://github.com/huplay/GPT-NEO-XL

Alternatively you can use the original source of the parameters: https://huggingface.co/EleutherAI/gpt-neo-1.3B

In the latter case download this file and unzip to the `parameters` folder: https://huggingface.co/EleutherAI/gpt-neo-1.3B/blob/main/pytorch_model.bin

This `pytorch_model.bin` file is a standard PK ZIP file. Possibly you have to rename it to `.zip` and extract the files under the `archive/data` folder.

You will find files named as 8 digit long numbers. You can create the mapping using the info in the `archive/data.pkl` pickle file.
