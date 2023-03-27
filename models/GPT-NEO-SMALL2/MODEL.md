# GPT-NEO-SMALL parameters

(EleutherAI GPT-NEO-125M)

Source of the parameters: https://huggingface.co/EleutherAI/gpt-neo-125M

Download and unzip the content of the following file to the `parameters` folder: https://huggingface.co/EleutherAI/gpt-neo-125M/blob/main/pytorch_model.bin

This `pytorch_model.bin` file is a standard PK ZIP file. Possibly you have to rename it to `.zip` and extract the files under the `archive/data` folder.

You will find files named as 8 digit long numbers which is mapped to the standard format in the `model.properties` file.
(The mapping was created using the info in the `archive/data.pkl` pickle file.)
