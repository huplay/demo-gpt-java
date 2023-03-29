package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

public class RotaryPositionEmbedder implements PositionEmbedder
{
    public RotaryPositionEmbedder(Settings settings, ParameterReader parameterReader)
    {
        // TODO: Maybe we have to store the max length
    }

    @Override
    public float[] addFixedPosition(float[] input, int pos)
    {
        return input;
    }

    @Override
    public float[] addRelativePosition(float[] input, int pos)
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
