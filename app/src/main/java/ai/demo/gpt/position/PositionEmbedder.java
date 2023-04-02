package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

public interface PositionEmbedder
{
    float[] addFixedPosition(float[] input, int pos);

    float[] addRelativePosition(float[] input, int pos);

    public static PositionEmbedder getInstance(Settings settings, ParameterReader parameterReader)
    {
        switch (settings.getPositionEncoder())
        {
            case "LEARNED": return new LearnedPositionEmbedder(settings, parameterReader);
            case "RoPE": return new RotaryPositionEmbedder(settings, parameterReader);
        }

        throw new RuntimeException("Unknown position embedding: " + settings.getTokenizer());
    }
}
