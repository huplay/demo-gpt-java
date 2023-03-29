# GPT-J-6B parameters

(EleutherAI GPT-J-6B)

---

Uses Rotary Position Embedding (RoPE), which isn't implemented here, so this model doesn't work atm.

Publication: https://arxiv.org/abs/2104.09864 (20 Apr 2021, Jianlin Su et al.)

Original implementation (RoFormer, 22 Mar 2021, Jianlin Su et al.): https://github.com/ZhuiyiTechnology/roformer

---

Pytorch implementation (29 Jun - 16 Aug 2021): https://github.com/lucidrains/rotary-embedding-torch

GPT-J-6B implementation (31 Aug 2021, using Hugging Face repo): https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/gptj/modeling_gptj.py

Hugging Face documentation (RoFormer): https://huggingface.co/docs/transformers/model_doc/roformer

---

Source of the parameters: https://huggingface.co/EleutherAI/gpt-j-6B

Download and unzip the content of the following file to the `parameters` folder: https://huggingface.co/EleutherAI/gpt-j-6B/blob/main/pytorch_model.bin

This `pytorch_model.bin` file is a standard PK ZIP file. Possibly you have to rename it to `.zip` and extract the files under the `archive/data` folder.

You will find files named starting from 0 to 284 which is mapped to the standard format in the `model.properties` file.
(The mapping was created using the info in the `archive/data.pkl` pickle file.)


