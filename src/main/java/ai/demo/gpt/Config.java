package ai.demo.gpt;

public class Config
{
    private final ModelType modelType;
    private final String parametersPath;
    private final Tokenizer tokenizer;
    private final int maxLength;
    private final int topK;

    public Config(ModelType modelType, String parametersPath, Tokenizer tokenizer, int maxLength, int topK)
    {
        this.modelType = modelType;
        this.parametersPath = parametersPath;
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
        this.topK = topK;
    }

    public ModelType getModelType()
    {
        return modelType;
    }

    public String getParametersPath()
    {
        return parametersPath;
    }

    public Tokenizer getTokenizer()
    {
        return tokenizer;
    }

    public int getMaxLength()
    {
        return maxLength;
    }

    public int getTopK()
    {
        return topK;
    }
}
