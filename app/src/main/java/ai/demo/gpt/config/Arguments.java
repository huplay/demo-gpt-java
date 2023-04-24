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
    private final boolean isCalculationOnly;

    public Arguments(String name, String path, int lengthLimit, int topK, boolean isCalculationOnly)
    {
        this.name = name;
        this.path = path;
        this.lengthLimit = lengthLimit;
        this.topK = topK;
        this.isCalculationOnly = isCalculationOnly;
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

    public boolean isCalculationOnly()
    {
        return isCalculationOnly;
    }
}