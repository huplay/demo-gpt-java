# GPT demo # Java

This is a demo application which implements the OpenAI's GPT-2 and GPT-3 artificial intelligence language models in Java, for learning purposes.

The goal is to demonstrate the decoder-only transformer architecture (without training), not to create an optimized application. 

TensorFlow or similar tools are NOT used, everything is implemented here.

(GPT-1 isn't implemented because of the different tokenization and normalization order.)

## Models ##

### OpenAI GPT-2 ###

The GPT-2 models were published completely. (Code and trained parameters as well.) You can try all of these using the demo-gpt-java app:

| Name         | Hidden size | Dec. no. | Head no. | Max. length | Size of params |                                                                                                    |
|--------------|------------:|---------:|---------:|------------:|---------------:|----------------------------------------------------------------------------------------------------|
| GPT-2 SMALL  |         768 |       12 |       12 |        1024 |          124 M | [Link](https://github.com/huplay/GPT2-SMALL)                                                       | 
| GPT-2 MEDIUM |        1024 |       24 |       16 |        1024 |          355 M | [Link](https://github.com/huplay/GPT2-MEDIUM)                                                      |
| GPT-2 LARGE  |        1280 |       36 |       20 |        1024 |          774 M | [Link](https://github.com/huplay/GPT2-LARGE)                                                       |
| GPT-2 XL     |        1600 |       48 |       25 |        1024 |        1,558 M | [Link1](https://github.com/huplay/GPT2-XL) <br /> [Link2](https://github.com/huplay/GPT2-XL-part2) |

### OpenAI GPT-3 ###

The GPT-3 algorithm is known (almost identical to GPT-2), this application has implemented it, but the parameters are not published, so you can't use these here:

| Name                       | Hidden size | Dec. no. | Head no. | Max. length |   Size of params | 
|----------------------------|------------:|---------:|---------:|------------:|-----------------:|
| GPT-3 SMALL                |         768 |       12 |       12 |        2048 |            124 M |
| GPT-3 MEDIUM               |        1024 |       24 |       16 |        2048 |            355 M |
| GPT-3 LARGE                |        1536 |       24 |       16 |        2048 |            759 M |
| GPT-3 XL                   |        2048 |       24 |       24 |        2048 |          1,314 M |
| GPT-3 ADA                  |        2560 |       32 |       32 |        2048 |          2,649 M |
| GPT-3 BABBAGE              |        4096 |       32 |       32 |        2048 |          6,654 M |
| GPT-3 CURIE                |        5140 |       40 |       40 |        2048 |         12,948 M |
| GPT-3 DAVINCI / GPT-3      |       12288 |       96 |       96 |        2048 |        174,591 M |
| GPT-3 DAVINCI v2 / GPT-3.5 |       12288 |       96 |       96 |        4000 |        174,591 M |
| GPT-3 DAVINCI v3 / ChatGPT |       12288 |       96 |       96 |        4000 |        174,591 M |

### EleutherAI GPT ###

`EleutherAI` is attempting to recreate all the GPT-3 variants, training them on their own dataset (`Pile`). (https://www.eleuther.ai)

They are published the following models so far: 

| Name                               | Hidden size | Dec. no. | Head no. | Max. length | Size of params |                                                                                                             |
|------------------------------------|------------:|---------:|---------:|------------:|---------------:|-------------------------------------------------------------------------------------------------------------|
| GPT-NEO-SMALL <br /> GPT-NEO-125M  |         768 |       12 |       12 |        2048 |          124 M |  [Link](https://github.com/huplay/GPT-NEO-SMALL)                                                            |
| GPT-NEO-MEDIUM <br /> GPT-NEO-350M |        1024 |       24 |       16 |        2048 |          355 M |  [Link](https://github.com/huplay/GPT-NEO-MEDIUM)                                                           |
| GPT-NEO-XL <br /> GPT-NEO-1.3B     |        2048 |       24 |       16 |        2048 |        1,314 M |  [Link](https://github.com/huplay/GPT-NEO-XL)                                                               |
| GPT-NEO-ADA <br /> GPT-NEO-2.7B    |        2560 |       32 |       20 |        2048 |        2,649 M |  [Link1](https://github.com/huplay/GPT-NEO-ADA) <br /> [Link2](https://github.com/huplay/GPT-NEO-ADA-part2) |
| GPT-J-6B                           |        4096 |       28 |       16 |        2048 |        5,849 M |                                                                                                             |
| GPT-NEOX-20B                       |        6144 |       44 |       64 |        2048 |       20,250 M |                                                                                                             |


### BLOOM (Hugging Face) ###

BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) was created by over a thousand AI developers, organized by Hugging Face, published in May 2022.

It has a larger vocabulary (250,880 tokens) compared to the GPT-2/3 models (50,257), so the size of the word token embedding parameters is bigger. (More parameters as a GPT-2 model with the same architectural size.)

| Name       | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|------------|------------:|---------:|---------:|------------:|---------------:|
| BLOOM-560M |        1024 |       24 |       16 |        2048 |          561 M | 
| BLOOM-1b1  |        1536 |       24 |       16 |        2048 |        1,068 M |
| BLOOM-1b7  |        2048 |       24 |       16 |        2048 |        1,726 M |
| BLOOM-3B   |        2560 |       30 |       32 |        2048 |        3,007 M |
| BLOOM-7b1  |        4096 |       30 |       32 |        2048 |        7,077 M |
| BLOOM      |       14336 |       70 |      112 |        2048 |      176,276 M |


### LLaMA (Meta AI) ###

LLaMA (Large Language Model Meta AI) is a large language model announced by Meta (Facebook) in 23 Feb 2023. The trained parameters were shared only to academic researchers, but on 2 March it was leaked to the public.

Vocabulary size is 32,000.

| Name      | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-----------|------------:|---------:|---------:|------------:|---------------:|
| LLaMA-7B  |        4096 |       32 |       32 |        2048 |        6,583 M | 
| LLaMA-13B |        5120 |       40 |       40 |        2048 |       12,759 M |
| LLaMA-33B |        6656 |       60 |       52 |        2048 |       32,129 M |
| LLaMA-65B |        8192 |       80 |       64 |        2048 |       64,711 M |


### OpenAI GPT-4 ###

GPT-4 was released in 14 March 2023, but almost all technical details are kept secret. It is known this is a Transformer architecture, and as input it can accept images as well. (The output is purely text.)

| Name      | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-----------|------------:|---------:|---------:|------------:|---------------:|
| GPT-4-8k  |          ?? |       ?? |       ?? |        8096 |           ?? M | 
| GPT-4-32k |          ?? |       ?? |       ?? |       32768 |           ?? M |


## Install ##

1. Install Java (version 1.8 or above).


2. Download and unzip this module: https://github.com/huplay/demo-gpt-java

   (Or using git: ```git clone https://github.com/huplay/demo-gpt-java.git```)


3. Download and unzip the files with the trained parameters for the version you want to use.

    Because of the GitHub repo size limit, the parameters for the larger models stored in multiple repos. Copy these into a single folder.


4. Using a command line tool (`cmd`) enter into the main directory:
   
    ```cd demo-gpt-java```


5. Compile (build) the application:

   ```compile``` (On Windows)

   Or alternatively (after Maven install): ```mvn clean install```

## Execution ##

Execute the application:
```run < path-of-the-parameters >``` (On Windows)
    
Or on any systems:```java -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.App < path-of-the-parameters >``` 
  
Using larger models it is necessary to increase the heap size (memory for Java). The ```run.bat``` handles it automatically, but if the app is called directly you should use the Java -Xmx and Xms flags. 

## Additional command line parameters ##

- ``maxlength`` - Maximum number of generated tokens
- ``topk`` - Number of possibilities to chose from as next token

## Usage ##

The app shows a prompt, where you can provide a text:

```Input text:```

You can leave it empty, or type something, which will be continued by the system. If the maximum length is reached, or the response finished by an end-of-text token, a new prompt will be given.

Normally every prompt starts a completely new session (the state is cleared), but if you want to remain in the same context, start your input text by `//`.

To quit press Ctrl + C.

## Trained parameters ##

The trained parameters are stored in separate repositories, collected from the published dataset. These can be found in different formats, sometimes a serialized object from TensorFlow, sometimes a Pytorch binary, I had to convert these from Python to Java binary format.

The file format is the simplest you can imagine: the `.dat` files contain the series of big endian float values (4 bytes each).

Some files (`wte.dat`) were too big to upload to GitHub, so these are split into multiple files (.part1, .part2, ...), merged automatically when the parameters are read.

The files are placed into folders:
 - `input`: Parameter files used at the input side of the Transformer (token and position embedding)
 - `decoder1..n`: Parameter files for the actual decoder
 - `output`: Parameters used at the output side of the Transformer (final normalization)
 - `tokenizer`: Files used by the tokenizer
 - `setup`: It contains command files to set the necessary memory size for the actual model

Every dataset should contain a model.properties file, with the following entries:
 - `token.count`: number of tokens
 - `end.of.text.token`: token id for marking the END-OF-TEXT
 - `max.length`: context size, the number of tokens the system can process (limited by the position embedding)
 - `hidden.size`: the size of the hidden state
 - `decoder.count`: number of decoders
 - `attention.head.count`: number of attention heads
 - `attention.score.dividend`: dividend at attention scoring (usually the square root of embeddingSize / headCount)
 - `attention.type.n`: attention type for all decoders (global | local | none)
 - `epsilon`: epsilon, used at normalization (mostly 1e-5f)

Optional properties:
 - `name`: name of the model
 - `source`: original creator of the model
 - `attention.local.size`: local attention size (necessary only if local attention type is used)
 - `hasAttentionQueryBias`: is there a bias for the attention query (default: true)
 - `hasAttentionKeyBias`: is there a bias for the attention key (default: true)
 - `hasAttentionValueBias`: is there a bias for the attention value (default: true)
 - `hasAttentionProjectionBias`: is there a bias for the attention projection (default: true)
 - `hasMlpLayer1Bias`: is there a bias for the mlp layer1 (default: true)
 - `hasMlpLayer2Bias`: is there a bias for the mlp layer2 (default: true)

## History ##

### Transformer ###

- Attention Is All You Need (2017, Google Brain)
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Usykoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- https://arxiv.org/abs/1706.03762 
- https://arxiv.org/pdf/1706.03762.pdf

This publication described an encoder-decoder Transformer architecture, optimal for translation between two languages.
The encoder stack creates an inner representation of the input language, the decoder stack transforms this representation to an output in the another language.
(The query, key and value vectors, created by the encoders are passed to the decoders.)
It was implemented using 6 encoders and 6 decoders.

### Decoder-only transformer ###

- Generating Wikipedia by summarizing long sentences (2018, Google Brain)
- Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Łukasz Kaiser, Noam Shazeer
- https://arxiv.org/pdf/1801.10198.pdf

This is a decoder-only variant of the Transformer architecture for natural language modeling. 
The decoder stack creates the query, key and value vectors (similarly as an encoder), without the input from the encoder stack (as there are no encoders).

### GPT-1 ###

- Improving Language Understanding by Generative Pre-Training (2018, OpenAI)
- Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- https://openai.com/blog/language-unsupervised/
- https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- Source code: https://github.com/openai/finetune-transformer-lm

OpenAI created a decoder-only implementation with 12 decoders and 12 heads. 
Instead of the originally proposed sinusoid position embedding it uses a trained position embedding matrix.

### GPT-2 ###

- Language Models are Unsupervised Multitask Learners (2019, OpenAI)
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- https://openai.com/blog/better-language-models/
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Source code: https://github.com/openai/ai.demo.gpt-2

GPT-2 has four variants: The smallest has the same size as the GPT-1 (12 decoders, 12 heads), the largest (XL) has 48 decoders and 25 heads.

The only architectural change to the GPT-1 is that the normalization within the decoders are moved before the attention and feed forward layers, and a final normalization is added after the last decoder.
(Instead of att/norm/add/mlp/norm/add it uses norm/att/add/norm/mlp/add steps.)

### Sparse Transformer ###

(This isn't implemented here.)

- Generating Long Sequences with Sparse Transformers (2019, OpenAI)
- Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever
- https://openai.com/blog/sparse-transformer/
- https://arxiv.org/pdf/1904.10509.pdf
- Source code: https://github.com/openai/sparse_attention

Proposal for a more efficient but still good performing sparse solution, where every second decoder uses a simplified calculation.

### GPT-3 ###

(This isn't implemented here.)

- Language Models are Few-Shot Learners (2020, OpenAI)
- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei
- https://arxiv.org/abs/2005.14165
- https://arxiv.org/pdf/2005.14165v4.pdf
- https://paperswithcode.com/paper/language-models-are-few-shot-learners/review/
- Source code (not complete): https://github.com/openai/ai.demo.gpt-3

Almost exactly the same architecture as GPT-2, but in different sizes, and some decoders use sparse attention. (The original attention called `global`, the new solution is `local`.) It is a very simple change: only the most recent tokens are used by the attention mechanism (last 128), the older tokens are dropped from the calculation.

### Tokenizer ###

All GPT versions use a byte pair encoding logic, but GPT-1 is different to the others.

GPT-1 had two types of tokens: a normal and a token at the end of the word (marked with `</w>`). Contrary, GPT-2 and GPT-3 have tokens containing an initial space.

GPT-1 used 40,000 merges, and 2 * 238 characters (normal and end-of-word variant), plus an `<unk>` (unknown) token.

GPT-2 and GPT-3 use 50,000 merges and 256 characters, plus an `end-of-text` token.


## Read more ##

Jay Alammar: 
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- The Illustrated GPT-2: https://jalammar.github.io/illustrated-gpt2
- How GPT-3 Works: https://jalammar.github.io/how-gpt3-works-visualizations-animations/

