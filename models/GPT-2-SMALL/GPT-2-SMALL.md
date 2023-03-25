# Parameters of the GPT-2 SMALL model
for the gpt-demo app

See: https://github.com/huplay/demo-gpt-java

Original source: https://github.com/openai/gpt-2
Parameter files can be downloaded using download_model.py,
which points to: https://openaipublic.blob.core.windows.net/gpt-2/models/124M/<fileName>

<fileName>:
    - checkpoint
    - encoder.json
    - hparams.json
    - model.ckpt.data-00000-of-00001
    - model.ckpt.index
    - model.ckpt.meta
    - vocab.bpe

From the above we needed the `model.ckpt.data-00000-of-00001` file, which contains the 

These parameters were extracted to the following repo: https://github.com/huplay/GPT2-SMALL
After downloading these files this model can be used by the demo-gpt app
