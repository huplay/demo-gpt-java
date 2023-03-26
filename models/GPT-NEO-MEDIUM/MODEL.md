# GPT-NEO-MEDIUM parameters

(EleutherAI GPT-NEO-350M)

You can download the parameter files from this repo: https://github.com/huplay/GPT-NEO-MEDIUM

Alternatively you can use the original source of the parameters: https://huggingface.co/xhyi/PT_GPTNEO350_ATG

In the latter case download this file and unzip to the `parameters` folder: https://huggingface.co/xhyi/PT_GPTNEO350_ATG/blob/main/pytorch_model.bin

This `pytorch_model.bin` file is a standard PK ZIP file. Possibly you have to rename it to `.zip` and extract the files under the `archive/data` folder.

You will find files named as 8 digit long numbers. You can create the mapping using the info in the `archive/data.pkl` pickle file.
