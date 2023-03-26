# GPT demo app

This is a demo application which implements the GPT-2/3 and similar artificial intelligence language models in Java, for learning purposes.

The goal is to demonstrate the decoder-only transformer architecture (without training), not to create an optimized application. (Larger models are very slow.)

TensorFlow, Pytorch or similar tools are NOT used, everything is implemented here. (There are separate branches using ND4j or Java Vector API.)


## Trained parameters ##

To use this app you have to find the trained parameters. For the GPT-2 models I stored these in separate repos what you can download. Other models are also ported, but you have to download them from their original repo.

There is a `models` folder where all the ported models have a subfolder with a configuration file. I also added a document (`MODEL.md`) which contains the details how to download that model.

For the details how to port a model see `MODELS.md` under the `models` folder.


## Install ##

1. Install Java (version 1.8 or above) and Maven.


2. Download and unzip this module: https://github.com/huplay/demo-gpt-java

   (Or using git: ```git clone https://github.com/huplay/demo-gpt-java.git```)


3. Download and unzip the files with the trained parameters for the version you want to use.

    Because of the GitHub repo size limit, the parameters for the larger models stored in multiple repos. Copy these into a single folder.


4. Using a command line tool (`cmd`) enter into the main directory:
   
    ```cd demo-gpt-java```


5. Compile (build) the application:

   ```mvn clean install```


## Execution ##

Execute the application:
```run < model-name >``` (On Windows)
    
Or on any systems:```java -cp target/demo-gpt-java-1.0.jar ai.demo.gpt.App < model-name >``` 
  
Using larger models it is necessary to increase the heap size (memory for Java). The ```run.bat``` handles it automatically, but if the app is called directly you should use the Java -Xmx and Xms flags. 


## Additional command line parameters ##

- `path` - Path of the `models` folder (default: `/models`)
- `maxlength` - Maximum number of generated tokens (default 25)
- `topk` - Number of possibilities to chose from as next token


## Usage ##

The app shows a prompt, where you can provide a text:

```Input text:```

You can leave it empty, or type something, which will be continued by the system. If the maximum length is reached, or the response finished by an end-of-text token, a new prompt will be given.

Normally every prompt starts a completely new session (the state is cleared), but if you want to remain in the same context, start your input text by `//`.

To quit press Ctrl + C.


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

