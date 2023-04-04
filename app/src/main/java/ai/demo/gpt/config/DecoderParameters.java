package ai.demo.gpt.config;

import ai.demo.util.Util;

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

        String decoder = "decoder" + (decoderId + 1) + "/";
        int hiddenSize = settings.getHiddenSize();
        int mlpSize = settings.getFeedForwardSize();

        // Don't load the weights of the first <memorySaverDecoders> decoders into memory if we use memory saver decoders
        boolean loadWeightsIntoMemory = decoderId >= settings.getMemorySaverDecoders();

        if (settings.isQueryKeyValueMerged())
        {
            if (loadWeightsIntoMemory)
            {
                // If the query, key and value matrices stored in the same file we have to split them
                float[][] qkvWeights = reader.readWeights(decoder + "att.query.key.value.w", hiddenSize * 3, hiddenSize);
                float[][][] qkvWeightSplit = Util.splitMatrix(qkvWeights, 3);

                this.queryWeights = qkvWeightSplit[0];
                this.keyWeights = qkvWeightSplit[1];
                this.valueWeights = qkvWeightSplit[2];
            }
            else
            {
                this.queryWeights = null;
                this.keyWeights = null;
                this.valueWeights = null;
            }

            float[] qkvBiases = reader.readVector(decoder + "att.query.key.value.b", hiddenSize * 3);
            float[][] qkvBiasSplit = Util.splitVector(qkvBiases, 3);

            this.queryBiases = qkvBiasSplit[0];
            this.keyBiases = qkvBiasSplit[1];
            this.valueBiases = qkvBiasSplit[2];
        }
        else
        {
            // Read the query, key and value matrices from separate files
            if (loadWeightsIntoMemory)
            {
                this.queryWeights = reader.readWeights(decoder + "att.query.w", hiddenSize, hiddenSize);
                this.keyWeights = reader.readWeights(decoder + "att.key.w", hiddenSize, hiddenSize);
                this.valueWeights = reader.readWeights(decoder + "att.value.w", hiddenSize, hiddenSize);
            }
            else
            {
                this.queryWeights = null;
                this.keyWeights = null;
                this.valueWeights = null;
            }

            this.queryBiases = reader.readVector(decoder + "att.query.b", hiddenSize);
            this.keyBiases = reader.readVector(decoder + "att.key.b", hiddenSize);
            this.valueBiases = reader.readVector(decoder + "att.value.b", hiddenSize);
        }

        if (loadWeightsIntoMemory)
        {
            this.projectionWeights = reader.readWeights(decoder + "att.proj.w", hiddenSize, hiddenSize);
            this.mlpLayer1Weights = reader.readWeights(decoder + "mlp.layer1.w", mlpSize, hiddenSize);
            this.mlpLayer2Weights = reader.readWeights(decoder + "mlp.layer2.w", hiddenSize,mlpSize);
        }
        else
        {
            this.projectionWeights = null;
            this.mlpLayer1Weights = null;
            this.mlpLayer2Weights = null;
        }

        this.projectionBiases = reader.readVector(decoder + "att.proj.b", hiddenSize);
        this.attNormWeights = reader.readVector(decoder + "att.norm.w", hiddenSize);
        this.attNormBiases = reader.readVector(decoder + "att.norm.b", hiddenSize);
        this.mlpLayer1Biases = reader.readVector(decoder + "mlp.layer1.b", mlpSize);
        this.mlpLayer2Biases = reader.readVector(decoder + "mlp.layer2.b", hiddenSize);
        this.mlpNormWeights = reader.readVector(decoder + "mlp.norm.w", hiddenSize);
        this.mlpNormBiases = reader.readVector(decoder + "mlp.norm.b", hiddenSize);
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
