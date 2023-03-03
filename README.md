# GPT demo # Java

This is demo application which implements the OpenAI's GPT-2 and GPT-3 artificial intelligence language models in Java, for learning purposes.

The goal is to demonstrate the decoder-only transformer architecture (without training), not to create an optimized application. 

TensorFlow or similar tools are NOT used, everything is implemented here.

## Install ##

1. Install Java (version 1.8 or above), optionally Maven and Git


2. Download and unzip or `git clone` this source module: https://github.com/huplay/demo-gpt-java

    ```git clone https://github.com/huplay/demo-gpt-java.git```


3. Download the parameter files with the trained parameters for the version you want to use. The default path is the `/parameters` folder, so the simplest way copying these files under that folder, but you can configure it differently.   
- GPT-2 SMALL: https://github.com/huplay/gpt2-ai.demo-small-params
- GPT-2 MEDIUM: https://github.com/huplay/gpt2-ai.demo-medium-params
- GPT-2 LARGE: https://github.com/huplay/gpt2-ai.demo-large-params
- GPT-2 XL: https://github.com/huplay/gpt2-ai.demo-xl-params1, https://github.com/huplay/gpt2-ai.demo-xl-params2

   The original OpenAI GPT-3 parameters are not available, but the EleutherAI trained some GPT-3 models.  

4. Using a command line tool (`cmd`) enter into the main directory:
   
    ```cd demo-gpt-java```


5. Compile (build) the application:

   ```compile.bat``` (On Windows)

   Or alternatively (after Maven install): ```mvn clean install```

## Usage ##

Execute the application:
```run.bat``` (On Windows, small version)
   
(Alternatively: ```run-medium.bat```, ```run-large.bat``` or ```run-xl.bat``` for the larger GPT-2 models)
    
Or on any systems:```java -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.Application model=SMALL``` (small version)

   
The app shows a prompt, where you can provide a starting text which will be continued by the app:

```Input text:```

(If you want to quit, type a single `q` and press Enter.)

The prompt will be repeated, but it will start a completely new session every time. (This isn't for chatting.)

Make sure you have enough memory for the particular model type. (A recommended value is specified in the .bat files using the -Xmx and -Xms flags.)

## Command line parameters ##

- ``model`` - Model size: SMALL (default), MEDIUM, LARGE, XL for the GPT-2, for GPT-3 models see `ModelType.java`
- ``path`` - Path of the parameter files (default: /parameters) 
- ``maxlength`` - Maximum number of generated tokens
- ``topk`` - Number of possibilities to chose from as next token

## Trained parameters ##

The trained data collected using the original GPT-2 application, converted into Java binary format.

Source: https://github.com/openai/ai.demo.gpt-2

The file format is the simplest, the `.dat` files contain the series of big endian float values (4 bytes each).

The `wte` files were too big to upload to GitHub, so these are split into multiple files (.part1, .part2, ...), merged automatically when the parameters are read.


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
(So instead of att/add/norm/mlp/add/norm it uses norm/att/add/norm/mlp/add steps.)

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

"Same model and architecture as GPT-2, including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer.
To study the dependence of ML performance on model size, we train 8 different sizes of model, from 125 million parameters to 175 billion parameters, with the last being the model we call GPT-3."

## Read more ##

Jay Alammar: 
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- The Illustrated GPT-2: https://jalammar.github.io/illustrated-gpt2
- How GPT-3 Works: https://jalammar.github.io/how-gpt3-works-visualizations-animations/

