package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

public interface PositionEmbedder
{
    float[] applyToInput(float[] input, int pos);

    float[] applyToQuery(float[] input, int length, int pos, int head);

    float[] applyToKey(float[] input, int length, int pos, int head);

    float[] applyToValue(float[] input, int length, int pos, int head);

    float applyToScore(float input, int length, int pos, int head);

    public static PositionEmbedder getInstance(Settings settings, ParameterReader parameterReader)
    {
        switch (settings.getPositionEncoder())
        {
            case "LEARNED": return new LearnedPositionEmbedder(settings, parameterReader);
            case "SINUSOIDAL": return new SinusoidalPositionEmbedder(settings, parameterReader);
            case "ROPE": return new RotaryPositionEmbedder(settings, parameterReader);
            case "ALIBI": return new AlibiPositionEmbedder(settings, parameterReader);
        }

        throw new RuntimeException("Unknown position embedding: " + settings.getTokenizer());
    }
}
