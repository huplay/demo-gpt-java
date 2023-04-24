# GPT-NEO #

`EleutherAI` is attempting to recreate all the GPT-3 variants, training them on their own dataset (`Pile`). (https://www.eleuther.ai)

They started the work on the smallest models, which are similar in size to the GPT-2.
I called the GPT-2 models as SMALL/MEDIUM/LARGE/XL. They officially released a model similar to the SMALL (NEO-125M) and XL (NEO-1.3B).
They trained a MEDIUM model as well (NEO-350M), which is mentioned occasionally, but it isn't uploaded to the official page. (I found only a copy uploaded by someone else.)
After these they've done a bigger model which is similar in size to the GPT-3 Ada (NEO-2.7B). (That is the smallest GPT-3 model that is available for the public.)

These above-mentioned models are under the NEO series.
(Later they modified their code base and some implementation details, so the larger models are under the GPT-J and GPT-NEOX series.)


| Name                               | Hidden size | Dec. no. | Head no. | Max. length | Size of params |                                                        |
|------------------------------------|------------:|---------:|---------:|------------:|---------------:|--------------------------------------------------------|
| GPT-NEO-SMALL <br /> GPT-NEO-125M  |         768 |       12 |       12 |        2048 |          124 M | [Link](https://huggingface.co/EleutherAI/gpt-neo-125M) |
| GPT-NEO-MEDIUM <br /> GPT-NEO-350M |        1024 |       24 |       16 |        2048 |          355 M | [Link](https://huggingface.co/xhyi/PT_GPTNEO350_ATG)   |
| GPT-NEO-XL <br /> GPT-NEO-1.3B     |        2048 |       24 |       16 |        2048 |        1,314 M | [Link](https://huggingface.co/EleutherAI/gpt-neo-1.3B) |
| GPT-NEO-ADA <br /> GPT-NEO-2.7B    |        2560 |       32 |       20 |        2048 |        2,649 M | [Link](https://huggingface.co/EleutherAI/gpt-neo-2.7B) |


The GPT-NEO series has the following specialities comparing to the GPT-2 models:
- Every second decoder uses sparse (local) attention. (This is the only architectural difference between GPT-2 and GPT-3 by the way.) 
- BFLOAT16 data type is used for the parameters, so instead of 4 bytes it uses only 2. It means the numbers are not as precise, but you can save memory.
  But the files itself stores 4 bytes for each number, where the first two bytes are fixed zeros.
  It means these are in fact FLOAT32 values, but the numbers not as precise how it could be.
  Because there are no 16 bit float variable type in Java I used these numbers as 32 bit float values,
  so the calculation isn't exactly the same as using 16 bit arithmetic, and no memory saving, but it works. 
- There are no biases for the attention query/key/value matrices. That's why I had to make the bias optional.
  (The file mapping contains `<NULL>` values for these files.)
- The attention dividend is missing, so the score isn't divided by a value usually calculated as square root of `hidden size` / `head count`.


Using the links in the table above you can download and unzip the content of the following file to the `parameters` folder: /blob/main/pytorch_model.bin

This `pytorch_model.bin` file is a standard PK ZIP file. Possibly you have to rename it to `.zip` and extract the files under the `archive/data` folder.

You will find files named as 8 or 9 digit long numbers.
(The mapping was created using the info in the `archive/data.pkl` pickle file.)


## Converted parameters ##

When I originally ported these models I converted the original files into BIG ENDIAN binary files, without investigating what was the original format.

The converted parameter files are stored in my repos:

| Name                               |                                                                                                            |
|------------------------------------|------------------------------------------------------------------------------------------------------------|
| GPT-NEO-SMALL <br /> GPT-NEO-125M  | [Link](https://github.com/huplay/GPT-NEO-SMALL)                                                            |
| GPT-NEO-MEDIUM <br /> GPT-NEO-350M | [Link](https://github.com/huplay/GPT-NEO-MEDIUM)                                                           |
| GPT-NEO-XL <br /> GPT-NEO-1.3B     | [Link](https://github.com/huplay/GPT-NEO-XL)                                                               |
| GPT-NEO-ADA <br /> GPT-NEO-2.7B    | [Link1](https://github.com/huplay/GPT-NEO-ADA) <br /> [Link2](https://github.com/huplay/GPT-NEO-ADA-part2) |

You can use these parameter files putting them into the GPT-NEO/converted/<model name> folder and start the app providing this path.


