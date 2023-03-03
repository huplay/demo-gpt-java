package ai.demo.gpt;

public enum ModelType
{
    // GPT-2 models:
    SMALL(50257, 1024, 768, 12, 12, false), // 124M
    MEDIUM(50257, 1024, 1024, 24, 16, false), // 355M
    LARGE(50257, 1024, 1280, 36, 20, false), // 774M
    XL(50257, 1024, 1600, 48, 25, false), // 1558M

    // TODO: Sparse model isn't implemented yet; no available trained parameters
    // GPT-3 models:
    GPT3_SMALL(50257, 2048, 768, 12, 12, true),
    GPT3_MEDIUM(50257, 2048, 1024, 24, 16, true),
    GPT3_LARGE(50257, 2048, 1536, 24, 16, true),
    GPT3_XL(50257, 2048, 2048, 24, 24, true),
    GPT3_ADA(50257, 2048, 2560, 32, 32, true),
    GPT3_BABBAGE(50257, 2048, 4096, 32, 32, true),
    GPT3_CURIE(50257, 2048, 5140, 40, 40, true),
    GPT3_DAVINCI(50257, 2048, 12288, 96, 96, true), // GPT-3
    GPT3_DAVINCI_V2(50257, 4000, 12288, 96, 96, true), // GPT-3.5
    GPT3_DAVINCI_V3(50257, 4000, 12288, 96, 96, true); // ChatGPT

    public final int tokenCount;
    public final int contextSize;
    public final int embeddingSize;
    public final int decoderCount;
    public final int headCount;
    public final boolean isSparse;

    ModelType(int tokenCount, int contextSize, int embeddingSize, int decoderCount, int headCount, boolean isSparse)
    {
        this.tokenCount = tokenCount;
        this.contextSize = contextSize;
        this.embeddingSize = embeddingSize;
        this.decoderCount = decoderCount;
        this.headCount = headCount;
        this.isSparse = isSparse;
    }

    public static ModelType find(String name)
    {
        ModelType modelType = SMALL;

        try
        {
            modelType = ModelType.valueOf(name.toUpperCase());
        }
        catch (IllegalArgumentException e)
        {
            Application.OUT.println("\nWARNING: The selected model type does not exist (" + name + ").\n");
        }

        return modelType;
    }

    public long getParameterSize()
    {
        long wteSize = (long) tokenCount * embeddingSize;
        long wpeSize = (long) contextSize * embeddingSize;
        long finalNormSize = (long) embeddingSize * 2;

        return wteSize + wpeSize + (getDecoderParameterSize() * decoderCount) + finalNormSize;
    }

    private long getDecoderParameterSize()
    {
        long qkvSize = ((long) embeddingSize * embeddingSize + embeddingSize) * 3;
        long projSize = (long) embeddingSize * embeddingSize + embeddingSize;
        long normSize = (long) embeddingSize * 4;
        long layer1Size = ((long) embeddingSize * embeddingSize + embeddingSize) * 4;
        long layer2Size = (long) embeddingSize * embeddingSize * 4 + embeddingSize;

        return qkvSize + projSize + normSize + layer1Size + layer2Size;
    }
}
