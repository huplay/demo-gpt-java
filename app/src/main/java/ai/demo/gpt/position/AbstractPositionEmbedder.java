package ai.demo.gpt.position;

public class AbstractPositionEmbedder implements PositionEmbedder
{
    @Override
    public float[] applyToInput(float[] input, int pos)
    {
        return input;
    }

    @Override
    public float[] applyToQuery(float[] input, int length, int pos, int head)
    {
        return input;
    }

    @Override
    public float[] applyToKey(float[] input, int length, int pos, int head)
    {
        return input;
    }

    @Override
    public float[] applyToValue(float[] input, int length, int pos, int head)
    {
        return input;
    }

    @Override
    public float applyToScore(float input, int length, int pos, int head)
    {
        return input;
    }
}
