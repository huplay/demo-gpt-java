package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

public class RotaryPositionEmbedder extends AbstractPositionEmbedder
{
    public RotaryPositionEmbedder(Settings settings, ParameterReader parameterReader)
    {
        // TODO: Maybe we have to store the max length
    }

    @Override
    public float[] applyToQuery(float[] input, int length, int pos, int head)
    {
        // TODO: Implement the RoPE
        // https://arxiv.org/abs/2104.09864
        // https://huggingface.co/docs/transformers/model_doc/roformer
        // https://github.com/ZhuiyiTechnology/roformer/commits/main
        // https://github.com/lucidrains/rotary-embedding-torch/commits/main
        // https://nn.labml.ai/transformers/rope/index.html

        return input;
    }
}
