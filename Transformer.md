# The Transformer architecture #

This is the original Transformer architecture, described in the famous "Attention Is All You Need" paper (2017, Google Brain):

https://arxiv.org/pdf/1706.03762.pdf

<img src="./images/original encoder-decoder transformer.png" width="400px">

Some reason the input is at the bottom, which is a little confusing to me, so I flipped it:

<img src="./images/original encoder-decoder transformer flipped.png" width="400px">

This is an encoder-decoder transformer, but the GPT and similar language models use a decoder-only variant, which was described by the Google Brain team in 2018. Unfortunately there was no drawing in the paper, so even now it is more difficult to find a correct image about the decoder-only architecture, so I created one: 

https://arxiv.org/pdf/1801.10198.pdf

<img src="./images/Decoder-only transformer flipped.png" width="250px">

The place of the normalization ("Norm") at the originally proposed variants and for example at the GPT-1 were different to the later implementations. Originally the attention or feed forward step happened first, and the normalization second. It was swapped in GPT-2, and a final normalization was added after the last decoder. My image above shows the solution used at GPT-2 and GPT-3. I also modified displaying the "Add" step. In the first images this was merged with the normalization ("Add & Norm"), but here it is separated (using the "+" sign).

(This application implements the image above.) 



