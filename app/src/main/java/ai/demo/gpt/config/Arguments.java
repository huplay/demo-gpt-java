package ai.demo.gpt.config;

/**
 * Holder of the app's input parameters
 */
public class Arguments
{
    private final String name;
    private final String path;
    private final int lengthLimit;
    private final int topK;

    public Arguments(String name, String path, int lengthLimit, int topK)
    {
        this.name = name;
        this.path = path;
        this.lengthLimit = lengthLimit;
        this.topK = topK;
    }

    public String getName()
    {
        return name;
    }

    public String getPath()
    {
        return path;
    }

    public int getLengthLimit()
    {
        return lengthLimit;
    }

    public int getTopK()
    {
        return topK;
    }

    public String getModelPath()
    {
        return path + "/" + name;
    }
}