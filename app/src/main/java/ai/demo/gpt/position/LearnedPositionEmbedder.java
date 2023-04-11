package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;
import ai.demo.util.Util;

public class LearnedPositionEmbedder extends AbstractPositionEmbedder
{
    private final float[][] positionEmbeddings;

    public LearnedPositionEmbedder(Settings settings, ParameterReader parameterReader)
    {
        this.positionEmbeddings = parameterReader.readMatrix("input/wpe", settings.getMaxLength(), settings.getHiddenSize());
    }

    @Override
    public float[] applyToInput(float[] input, int pos)
    {
        return Util.addVectors(input, positionEmbeddings[pos]);
    }
}
