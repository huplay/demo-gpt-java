# BLOOM #

BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) was created by over a thousand AI developers, organized by Hugging Face, published in May 2022.

These models have the following specialities:
- Custom tokenizer method, using 250,880 tokens
- ALiBi position embedding
- Additional normalization after the input embedding step

| Name       | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|------------|------------:|---------:|---------:|------------:|---------------:|
| BLOOM-560M |        1024 |       24 |       16 |        2048 |          561 M | 
| BLOOM-1.1B |        1536 |       24 |       16 |        2048 |        1,068 M |
| BLOOM-1.7B |        2048 |       24 |       16 |        2048 |        1,726 M |
| BLOOM-3B   |        2560 |       30 |       32 |        2048 |        3,007 M |
| BLOOM-7.1B |        4096 |       30 |       32 |        2048 |        7,077 M |
| BLOOM      |       14336 |       70 |      112 |        2048 |      176,276 M |

You can download the trained parameters using the links above. Only the `pytorch_model.bin` file is needed. (This is a `PKZIP` file, just you have to rename it to `.zip` to make it obvious.)
You have to extract the files under the `archive/data` folder into the `parameters` folder.

You can find a direct link in the `README.md` file within the particular `parameters` folder.

(The mapping was created using the info in the `archive/data.pkl` pickle file.)
