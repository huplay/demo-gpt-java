package ai.demo.gpt.position;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;

/**
 * ALiBi position embedding (Attention Linear Bias) implementation
 * Publication: https://arxiv.org/abs/2108.12409
 * Used at the BLOOM models
 */
public class AlibiPositionEmbedder extends AbstractPositionEmbedder
{
    private final float[] slopes;

    public AlibiPositionEmbedder(Settings settings, ParameterReader parameterReader)
    {
        this.slopes = calculateSlopes(settings.getHeadCount());
    }

    private float[] calculateSlopes(int headCount)
    {
        float[] slopes = new float[headCount];

        float step = 1f / headCount;

        for (int i = 0; i < headCount; i++)
        {
            slopes[i] = step * (i + 1);
        }

        return slopes;
    }

    @Override
    public float toScore(float input, int length, int pos, int head)
    {
        return input - slopes[head] * (length - pos - 1);
    }
}
