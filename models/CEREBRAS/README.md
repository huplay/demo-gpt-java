# Cerebras #

Cerebras released seven GPT models on 28 March 2023 on the `Hugging Face` platform: https://huggingface.co/cerebras

The home page of the company: https://www.cerebras.net/

Official announcement of the release: https://www.cerebras.net/press-release/cerebras-systems-releases-seven-new-gpt-models-trained-on-cs-2-wafer-scale-systems

Architecturally these are very similar models to the OpenAI's GPT-2 series. Only global attention is used (no sparse), same tokenizer, same trained position embedding.
The first four models are all a little bit smaller than the four GPT-2 variants. (So the smallest is smaller than the GPT-1.) The three larger models are similar in size to the GPT-3 Ada, Babbage and Curie models.

For the training they used the `Pile` dataset, originally collected by the EleutherAI team: https://arxiv.org/abs/2101.00027


| Name          | Hidden size | Dec. no. | Head no. | Max. length | Size of params | Download                                                   |
|---------------|------------:|---------:|---------:|------------:|---------------:|------------------------------------------------------------|
| Cerebras-111M |         768 |       10 |       12 |        2048 |          111 M | [Link](https://huggingface.co/cerebras/Cerebras-GPT-111M)  |
| Cerebras-256M |        1088 |       14 |       17 |        2048 |          256 M | [Link](https://huggingface.co/cerebras/Cerebras-GPT-256M)  |
| Cerebras-590M |        1536 |       18 |       12 |        2048 |          590 M | [Link](https://huggingface.co/cerebras/Cerebras-GPT-590M)  |
| Cerebras-1.3B |        2048 |       24 |       16 |        2048 |        1,316 M | [Link](https://huggingface.co/cerebras/Cerebras-GPT-1.3B)  |
| Cerebras-2.7B |        2560 |       32 |       32 |        2048 |        2,652 M | [Link](https://huggingface.co/cerebras/Cerebras-GPT-2.7B)  |
| Cerebras-6.7B |        4096 |       32 |       32 |        2048 |        6,658 M | [Link](https://huggingface.co/cerebras/Cerebras-GPT-6.7B)  |
| Cerebras-13B  |        5120 |       40 |       40 |        2048 |       12,853 M | [Link](https://huggingface.co/cerebras/Cerebras-GPT-13B)   |

You can download the trained parameters using the links above. Only the `pytorch_model.bin` file is needed. (This is a `PKZIP` file, just you have to rename it to `.zip` to make it obvious.)
You have to extract the files under the `archive/data` folder into the `parameters` folder.

You can find a direct link in the `README.md` file within the particular `parameters` folder.

(The mapping was created using the info in the `archive/data.pkl` pickle file.)
