package ai.demo.gpt.config;

public class DecoderParameters
{
    private final int decoderId;
    private final Settings settings;
    private final ParameterReader reader;
    private boolean isAttentionInitialized;
    private boolean isMlpInitialized;

    private float[] attNormWeights;
    private float[] attNormBiases;
    private QueryKeyValueParameters queryKeyValueParameters;
    private float[][] projectionWeights;
    private float[] projectionBiases;

    private float[] mlpNormWeights;
    private float[] mlpNormBiases;
    private float[][] mlpLayer1Weights;
    private float[] mlpLayer1Biases;
    private float[][] mlpLayer2Weights;
    private float[] mlpLayer2Biases;

    public DecoderParameters(int decoderId, Settings settings, ParameterReader reader)
    {
        this.decoderId = decoderId;
        this.settings = settings;
        this.reader = reader;

        initAttentionBlock(false);
        initNeuronBlock(false);
    }

    public void initAttentionBlock(boolean isForced)
    {
        if ( ! isAttentionInitialized && (isForced || decoderId >= settings.getMemorySaverDecoders()))
        {
            int hiddenSize = settings.getHiddenSize();

            this.attNormWeights = reader.readVector(prefixedName("att.norm.w"), hiddenSize);
            this.attNormBiases = reader.readVector(prefixedName("att.norm.b"), hiddenSize);
            this.queryKeyValueParameters = new QueryKeyValueParameters(decoderId, settings, reader);
            this.projectionWeights = reader.readWeights(prefixedName("att.proj.w"), hiddenSize, hiddenSize);
            this.projectionBiases = reader.readVector(prefixedName("att.proj.b"), hiddenSize);

            this.isAttentionInitialized = true;
        }
    }

    public void closeAttentionBlock()
    {
        if (decoderId < settings.getMemorySaverDecoders())
        {
            this.attNormWeights = null;
            this.attNormBiases = null;
            this.queryKeyValueParameters = null;
            this.projectionWeights = null;
            this.projectionBiases = null;

            this.isAttentionInitialized = false;
        }
    }

    public void initNeuronBlock(boolean isForced)
    {
        if ( ! isMlpInitialized && (isForced || decoderId >= settings.getMemorySaverDecoders()))
        {
            int hiddenSize = settings.getHiddenSize();
            int feedForwardSize = settings.getFeedForwardSize();

            this.mlpNormWeights = reader.readVector(prefixedName("mlp.norm.w"), hiddenSize);
            this.mlpNormBiases = reader.readVector(prefixedName("mlp.norm.b"), hiddenSize);
            this.mlpLayer1Weights = reader.readWeights(prefixedName("mlp.layer1.w"), feedForwardSize, hiddenSize);
            this.mlpLayer1Biases = reader.readVector(prefixedName("mlp.layer1.b"), feedForwardSize);
            this.mlpLayer2Weights = reader.readWeights(prefixedName("mlp.layer2.w"), hiddenSize, feedForwardSize);
            this.mlpLayer2Biases = reader.readVector(prefixedName("mlp.layer2.b"), hiddenSize);

            this.isMlpInitialized = true;
        }
    }

    public void closeNeuronBlock()
    {
        if (decoderId < settings.getMemorySaverDecoders())
        {
            this.mlpNormWeights = null;
            this.mlpNormBiases = null;
            this.mlpLayer1Weights = null;
            this.mlpLayer1Biases = null;
            this.mlpLayer2Weights = null;
            this.mlpLayer2Biases = null;

            this.isMlpInitialized = false;
        }
    }

    private String prefixedName(String name)
    {
        return "decoder" + (decoderId + 1) + "/" + name;
    }

    public float[][] getQueryWeights()
    {
        return queryKeyValueParameters.getQueryWeights();
    }

    public float[] getQueryBiases()
    {
        return queryKeyValueParameters.getQueryBiases();
    }

    public float[][] getKeyWeights()
    {
        return queryKeyValueParameters.getKeyWeights();
    }

    public float[] getKeyBiases()
    {
        return queryKeyValueParameters.getKeyBiases();
    }

    public float[][] getValueWeights()
    {
        return queryKeyValueParameters.getValueWeights();
    }

    public float[] getValueBiases()
    {
        return queryKeyValueParameters.getValueBiases();
    }

    public float[][] getProjectionWeights()
    {
        return projectionWeights;
    }

    public float[] getProjectionBiases()
    {
        return projectionBiases;
    }

    public float[] getAttNormWeights()
    {
        return attNormWeights;
    }

    public float[] getAttNormBiases()
    {
        return attNormBiases;
    }

    public float[][] getMlpLayer1Weights()
    {
        return mlpLayer1Weights;
    }

    public float[] getMlpLayer1Biases()
    {
        return mlpLayer1Biases;
    }

    public float[][] getMlpLayer2Weights()
    {
        return mlpLayer2Weights;
    }

    public float[] getMlpLayer2Biases()
    {
        return mlpLayer2Biases;
    }

    public float[] getMlpNormWeights()
    {
        return mlpNormWeights;
    }

    public float[] getMlpNormBiases()
    {
        return mlpNormBiases;
    }
}
