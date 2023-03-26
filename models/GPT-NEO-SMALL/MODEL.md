# GPT-NEO-SMALL parameters

(EleutherAI GPT-NEO-125M)

You can download the parameter files from this repo: https://github.com/huplay/GPT-NEO-SMALL

Alternatively you can use the original source of the parameters: https://huggingface.co/EleutherAI/gpt-neo-125M

In the latter case download this file and unzip to the `parameters` folder: https://huggingface.co/EleutherAI/gpt-neo-125M/blob/main/pytorch_model.bin

This `pytorch_model.bin` file is a standard PK ZIP file. Possibly you have to rename it to `.zip` and extract the files under the `archive/data` folder.

You will find files named as 8 digit long numbers. The mapping to the standard format is added to the `model2.properties` file. (Created using the info in the `archive/data.pkl` pickle file.)
