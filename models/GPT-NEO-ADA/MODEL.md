# GPT-NEO-ADA parameters

(EleutherAI GPT-NEO-2.7B)

You can download the parameter files from these repos: 
 - https://github.com/huplay/GPT-NEO-ADA
 - https://github.com/huplay/GPT-NEO-ADA-part2

Alternatively you can use the original source of the parameters: https://huggingface.co/EleutherAI/gpt-neo-2.7B

In the latter case download this file and unzip to the `parameters` folder: https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/main/pytorch_model.bin

This `pytorch_model.bin` file is a standard PK ZIP file. Possibly you have to rename it to `.zip` and extract the files under the `archive/data` folder.

You will find files named as 8 digit long numbers. You can create the mapping using the info in the `archive/data.pkl` pickle file.
