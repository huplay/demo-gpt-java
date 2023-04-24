package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

import static ai.demo.gpt.App.UTIL;

public class LearnedPositionEmbedder extends AbstractPositionEmbedder
{
    private final float[][] positionEmbeddings;

    public LearnedPositionEmbedder(Settings settings, ParameterReader parameterReader)
    {
        this.positionEmbeddings = parameterReader.readMatrix("input/wpe", settings.getMaxLength(), settings.getHiddenSize());
    }

    @Override
    public float[] toInput(float[] input, int pos)
    {
        return UTIL.addVectors(input, positionEmbeddings[pos]);
    }
}
