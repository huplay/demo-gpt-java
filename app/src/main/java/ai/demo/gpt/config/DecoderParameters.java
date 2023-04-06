package ai.demo.gpt.config;

public class DecoderParameters
{
    private final int decoderId;
    private final Settings settings;
    private final ParameterReader reader;

    private final float[][] queryWeights;
    private final float[] queryBiases;
    private final float[][] keyWeights;
    private final float[] keyBiases;
    private final float[][] valueWeights;
    private final float[] valueBiases;
    private final float[][] projectionWeights;
    private final float[] projectionBiases;
    private final float[] attNormWeights;
    private final float[] attNormBiases;
    private final float[][] mlpLayer1Weights;
    private final float[] mlpLayer1Biases;
    private final float[][] mlpLayer2Weights;
    private final float[] mlpLayer2Biases;
    private final float[] mlpNormWeights;
    private final float[] mlpNormBiases;

    public DecoderParameters(int decoderId, Settings settings, ParameterReader reader)
    {
        this.decoderId = decoderId;
        this.settings = settings;
        this.reader = reader;

        this.queryWeights = readWeights("att.query.w", false);
        this.queryBiases = readQKVBiases("att.query.b");
        this.keyWeights = readWeights("att.key.w", false);
        this.keyBiases = readQKVBiases("att.key.b");
        this.valueWeights = readWeights("att.value.w", false);
        this.valueBiases = readQKVBiases("att.value.b");
        this.projectionWeights = readWeights("att.proj.w", false);
        this.projectionBiases = reader.readVector(prefixedName("att.proj.b"), settings.getHiddenSize());
        this.attNormWeights = reader.readVector(prefixedName("att.norm.w"), settings.getHiddenSize());
        this.attNormBiases = reader.readVector(prefixedName("att.norm.b"), settings.getHiddenSize());
        this.mlpLayer1Weights = readWeights("mlp.layer1.w", false);
        this.mlpLayer1Biases = reader.readVector(prefixedName("mlp.layer1.b"), settings.getFeedForwardSize());
        this.mlpLayer2Weights = readWeights("mlp.layer2.w", false);
        this.mlpLayer2Biases = reader.readVector(prefixedName("mlp.layer2.b"), settings.getHiddenSize());
        this.mlpNormWeights = reader.readVector(prefixedName("mlp.norm.w"), settings.getHiddenSize());
        this.mlpNormBiases = reader.readVector(prefixedName("mlp.norm.b"), settings.getHiddenSize());
    }

    public float[][] readWeights(String name, boolean isForced)
    {
        int hiddenSize = settings.getHiddenSize();
        int feedForwardSize = settings.getFeedForwardSize();

        switch (name)
        {
            case "att.query.w": return readQKV(name, 0, hiddenSize, hiddenSize, isForced);
            case "att.key.w": return readQKV(name, 1, hiddenSize, hiddenSize, isForced);
            case "att.value.w": return readQKV(name, 2, hiddenSize, hiddenSize, isForced);
            case "att.proj.w": return readWeights(name, hiddenSize, hiddenSize, isForced);
            case "mlp.layer1.w": return readWeights(name, feedForwardSize, hiddenSize, isForced);
            case "mlp.layer2.w": return readWeights(name, hiddenSize, feedForwardSize, isForced);
            default: throw new RuntimeException("Unknown weight type: " + name);
        }
    }

    private float[][] readQKV(String name, int index, int rows, int cols, boolean isForced)
    {
        if (isForced || decoderId >= settings.getMemorySaverDecoders())
        {
            int segments = settings.isQueryKeyValueMerged() ? 3 : 1;
            name = settings.isQueryKeyValueMerged() ? "att.query.key.value.w" : name;

            return reader.readWeights(prefixedName(name), rows, cols, segments, index);
        }

        return null;
    }

    private float[][] readWeights(String name, int rows, int cols, boolean isForced)
    {
        if (isForced || decoderId >= settings.getMemorySaverDecoders())
        {
            return reader.readWeights(prefixedName(name), rows, cols, 1, 0);
        }

        return null;
    }

    private float[] readQKVBiases(String name)
    {
        int segments = settings.isQueryKeyValueMerged() ? 3 : 1;
        String finalName = settings.isQueryKeyValueMerged() ? "att.query.key.value.b" : name;

        switch (name)
        {
            case "att.query.b": return reader.readVector(prefixedName(finalName), settings.getHiddenSize(), segments, 0);
            case "att.key.b": return reader.readVector(prefixedName(finalName), settings.getHiddenSize(), segments, 1);
            case "att.value.b": return reader.readVector(prefixedName(finalName), settings.getHiddenSize(), segments, 2);
            default : throw new RuntimeException("Unknown bias type: " + name);
        }
    }

    private String prefixedName(String name)
    {
        return "decoder" + (decoderId + 1) + "/" + name;
    }

    public float[][] getQueryWeights()
    {
        return queryWeights;
    }

    public float[] getQueryBiases()
    {
        return queryBiases;
    }

    public float[][] getKeyWeights()
    {
        return keyWeights;
    }

    public float[] getKeyBiases()
    {
        return keyBiases;
    }

    public float[][] getValueWeights()
    {
        return valueWeights;
    }

    public float[] getValueBiases()
    {
        return valueBiases;
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
