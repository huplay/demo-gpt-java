package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

public class SinusoidalPositionEmbedder extends AbstractPositionEmbedder
{
    public SinusoidalPositionEmbedder(Settings settings, ParameterReader parameterReader)
    {
        // TODO: Maybe we have to store the max length
    }

    @Override
    public float[] applyToInput(float[] input, int pos)
    {
        return input;
    }
}
