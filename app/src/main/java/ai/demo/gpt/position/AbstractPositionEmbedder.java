package ai.demo.gpt.position;

public class AbstractPositionEmbedder implements PositionEmbedder
{
    @Override
    public float[] toInput(float[] input, int pos)
    {
        return input;
    }

    @Override
    public float[] toQuery(float[] input, int length, int pos, int head)
    {
        return input;
    }

    @Override
    public float[] toKey(float[] input, int length, int pos, int head)
    {
        return input;
    }

    @Override
    public float[] toValue(float[] input, int length, int pos, int head)
    {
        return input;
    }

    @Override
    public float toScore(float input, int length, int pos, int head)
    {
        return input;
    }
}
