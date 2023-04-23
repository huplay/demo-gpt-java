package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

public interface PositionEmbedder
{
    float[] toInput(float[] input, int pos);

    float[] toQuery(float[] input, int length, int pos, int head);

    float[] toKey(float[] input, int length, int pos, int head);

    float[] toValue(float[] input, int length, int pos, int head);

    float toScore(float input, int length, int pos, int head);

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
