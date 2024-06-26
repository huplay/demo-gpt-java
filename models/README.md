# GPT demo parameters #

To use the GPT demo app you have to download the parameter files.

You can find a lot of ported models under the `model` folder, but the parameter files are missing, you have to download those.

For some models I've created a separate repo storing these parameter files. (It was necessary because the files were stored using a Tensorflow serialized format which isn't known by this app.) For other models it is possible to use the original files, so I just added a description how to collect them.

The parameter files should be stored under the `parameters` folder within the main folder of the model.

For every model it is necessary to have a `model.properties` file which describes the main configuration and the format of the parameter files.

The standard structure for the parameters is the following:
- `input`: Containing parameter files used at the input side of the Transformer (token and position embedding)
  - `wte.dat`, `wpe.dat`
- `decoder1..n`: Containing parameter files for the actual decoder
  - `att.query.w.dat`, `att.query.b.dat`, `att.key.w.dat`, `att.key.b.dat`, `att.value.w.dat`, `att.value.b.dat`,
    `att.proj.w.dat`, `att.proj.b.dat`, `att.norm.w.dat`, `att.norm.b.dat`
    `mlp.layer1.w.dat`, `mlp.layer1.b.dat`, `mlp.layer2.w.dat`, `mlp.layer2.b.dat`,
    `mlp.norm.w.dat`, `mlp.norm.b.dat`
- `output`: Containing parameters used at the output side of the Transformer (final normalization)
  - `norm.w.dat`, `norm.b.dat`

Some files are too big to upload to GitHub, so these are split into multiple files (`.part1`, `.part2`, ...). These files are merged automatically when the parameters are read. (The merged file is saved.)

If the parameters are from an external source using a different structure the mapping between the standard format and the actual can be described in the `model.properties` file.

Every model can contain a `setup.bat` file, which is used to configure the necessary memory size. (Adding a java parameter before the app is started.)

## Format of model.properties ##

- `tokenizer`: type of tokenizer (Currently only the GPT-2 is supported)
- `tokenizer.config`: the config folder of the tokenizer
- `token.count`: number of tokens
- `end.of.text.token`: token id for marking the END-OF-TEXT
- `max.length`: context size, the number of tokens the system can process (limited by the position embedding)
- `position.embedding`: the type of the position embedding (LEARNED, SINUSOIDAL, ROPE, ALIBI)
- `pre.normalization`: marks that, pre-normalization is used (opposing post-normalization) (true or false)
- `hidden.size`: the size of the hidden state
- `feedforward.size`: the size of the feed forward layer (usually four times the hidden size)
- `decoder.count`: number of decoders
- `attention.head.count`: number of attention heads
- `attention.dividend`: dividend at attention scoring - usually the square root of head size (hidden size / head count)
- `attention.type.n`: attention type for all decoders (global | local | none)
- `epsilon`: epsilon, used at normalization (mostly 1e-5f)
- `data.type`: data type of the values. (Currently, only FLOAT32 is supported: 4 bytes per value. FLOAT16 would be the another commonly used possibility.)
- `byte.order`: BIG_ENDIAN or LITTLE_ENDIAN
- `weights.transposed`: weights are stored horizontally (transposed) or not (vertically). Vertical is the more convenient order. (Used in Pytorch files.) 
- `file.<standard name>`: file mapping from the standard name. (Use <null> if the specific file is missing. It works only with bias files, the weight are compulsory.)

Optional properties:
- `name`: name of the model
- `source`: original creator of the model
- `attention.local.size`: local attention size (necessary only if local attention type is used)
- `memory.saver.decoders`: how many decoders use the weight parameters directly from disk, without fully loading into memory.
This slows down the process, but saves memory, so in theory you will be able to try larger models.

## Models ##

### GPT-2 (OpenAI) ###

The GPT-2 models were published completely. (Code and trained parameters as well.) You can try all of these using the demo-gpt app:

| Name         | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|--------------|------------:|---------:|---------:|------------:|---------------:|
| GPT-2 SMALL  |         768 |       12 |       12 |        1024 |          124 M |
| GPT-2 MEDIUM |        1024 |       24 |       16 |        1024 |          355 M |
| GPT-2 LARGE  |        1280 |       36 |       20 |        1024 |          774 M |
| GPT-2 XL     |        1600 |       48 |       25 |        1024 |        1,558 M |

### GPT-NEO/J/NEOX (EleutherAI) ###

`EleutherAI` is attempting to recreate all the GPT-3 variants, training them on their own dataset (`Pile`). (https://www.eleuther.ai)

I converted the parameters for the smaller models into my standard format, but optionally you can use the original Pytorch files as well. Here you can find the links for the converted parameters.

EleutherAI created the following models so far:

| Name                               | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|------------------------------------|------------:|---------:|---------:|------------:|---------------:|
| GPT-NEO-SMALL <br /> GPT-NEO-125M  |         768 |       12 |       12 |        2048 |          124 M |
| GPT-NEO-MEDIUM <br /> GPT-NEO-350M |        1024 |       24 |       16 |        2048 |          355 M |
| GPT-NEO-XL <br /> GPT-NEO-1.3B     |        2048 |       24 |       16 |        2048 |        1,314 M |
| GPT-NEO-ADA <br /> GPT-NEO-2.7B    |        2560 |       32 |       20 |        2048 |        2,649 M |
| GPT-J-6B                           |        4096 |       28 |       16 |        2048 |        5,849 M |
| GPT-NEOX-20B                       |        6144 |       44 |       64 |        2048 |       20,250 M |


### BLOOM ###

BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) was created by over a thousand AI developers, organized by Hugging Face, published in May 2022.

| Name       | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|------------|------------:|---------:|---------:|------------:|---------------:|
| BLOOM-560M |        1024 |       24 |       16 |        2048 |          559 M | 
| BLOOM-1.1B |        1536 |       24 |       16 |        2048 |        1,065 M |
| BLOOM-1.7B |        2048 |       24 |       16 |        2048 |        1,722 M |
| BLOOM-3B   |        2560 |       30 |       32 |        2048 |        3,003 M |
| BLOOM-7.1B |        4096 |       30 |       32 |        2048 |        7,069 M |
| BLOOM-176B |       14336 |       70 |      112 |        2048 |      176,247 M |


### LLaMA (Meta AI) ###

LLaMA (Large Language Model Meta AI) is a large language model announced by Meta (Facebook) in 23 Feb 2023. The trained parameters were shared only to academic researchers, but on 2 March it was leaked to the public.

Llama 2 was published in 18 July 2023 in three sizes. Llama 3 was published in 18 Apr 2024 in two sizes. (The training of a bigger version is in progress.)

The weights and the code is completely public for Llama 2 and 3, free to use even in commercial products. (Companies over 700 million monthly users has to request the licence.) 

These models are not ported yet as I don't have all the necessary details, but there's a chance the smaller models could be used somehow.

Llama 1 and 2 has a vocabulary of 32,000 tokens. 

| Name      | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-----------|------------:|---------:|---------:|------------:|---------------:|
| LLaMA-7B  |        4096 |       32 |       32 |        2048 |        6,583 M |
| LLaMA-13B |        5120 |       40 |       40 |        2048 |       12,759 M |
| LLaMA-33B |        6656 |       60 |       52 |        2048 |       32,129 M |
| LLaMA-65B |        8192 |       80 |       64 |        2048 |       64,711 M |

Llama 2 uses an additional gate projection network in the MLP block, and the 13B and 70B models has Grouped Query attention solution (GQA), so the calculation of the parameter size is a little bit different.

| Name        | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-------------|------------:|---------:|---------:|------------:|---------------:|
| Llama 2-7B  |        4096 |       32 |       32 |        4096 |              M |
| Llama 2-13B |        5120 |       40 |       40 |        4096 |       13,016 M |
| Llama 2-70B |        8192 |       80 |       64 |        4096 |              M |

Llama 3 architecturally the same as Llama 2, all models use GQA (even the smallest). It has a different tokenizer with 128,000 vocabulary size.
There's a base model and an instruct fine trained variant as well in each sizes.

| Name         | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|--------------|------------:|---------:|---------:|------------:|---------------:|
| Llama 3-8B   |        4096 |       32 |       32 |        8192 |              M |
| Llama 3-70B  |        8192 |       80 |       64 |        8192 |              M |
| Llama 3-400B |        ???? |       ?? |       ?? |        ???? |              M |

### Cerebras ###

`Cerebras` released seven models in March 2023, trained on `Pile`. (https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models)

These are very similar models to the GPT-2/GTP-3 series, using the same tokenizer, same learned position embedding, etc. Unlike GPT-3, using always global attention.

All of these are ported to this app:

| Name          | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|---------------|------------:|---------:|---------:|------------:|---------------:|
| Cerebras-111M |         768 |       10 |       12 |        2048 |          111 M |
| Cerebras-256M |        1088 |       14 |       17 |        2048 |          256 M |
| Cerebras-590M |        1536 |       18 |       12 |        2048 |          590 M |
| Cerebras-1.3B |        2048 |       24 |       16 |        2048 |         1316 M |
| Cerebras-2.7B |        2560 |       32 |       32 |        2048 |         2652 M |
| Cerebras-6.7B |        4096 |       32 |       32 |        2048 |         6658 M |
| Cerebras-13B  |        5120 |       40 |       40 |        2048 |        12853 M |


## Unpublished models ##

### GPT-3 (OpenAI) ###

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

### GPT-4 (OpenAI) ###

GPT-4 was released in 14 March 2023, but almost all technical details are kept secret. It is known this is a Transformer architecture, and as input it can accept images as well. (The output is purely text.)

| Name      | Hidden size | Dec. no. | Head no. | Max. length | Size of params |
|-----------|------------:|---------:|---------:|------------:|---------------:|
| GPT-4-8k  |          ?? |       ?? |       ?? |        8096 |           ?? M | 
| GPT-4-32k |          ?? |       ?? |       ?? |       32768 |           ?? M |
