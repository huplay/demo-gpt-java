package ai.demo.gpt.config;

import ai.demo.util.Util;

public class DecoderParameters
{
    private final int decoderId;
    private final Settings settings;
    private final ParameterReader reader;

    private float[][] queryWeights;
    private float[] queryBiases;
    private float[][] keyWeights;
    private float[] keyBiases;
    private float[][] valueWeights;
    private float[] valueBiases;
    private float[][] projectionWeights;
    private float[] projectionBiases;
    private float[] attNormWeights;
    private float[] attNormBiases;
    private float[][] mlpLayer1Weights;
    private float[] mlpLayer1Biases;
    private float[][] mlpLayer2Weights;
    private float[] mlpLayer2Biases;
    private float[] mlpNormWeights;
    private float[] mlpNormBiases;

    private boolean isLoaded = false;

    public DecoderParameters(int decoderId, Settings settings, ParameterReader reader)
    {
        this.decoderId = decoderId;
        this.settings = settings;
        this.reader = reader;

        if ( ! settings.isCleanDecoder(decoderId))
        {
            init();
        }
    }

    private void init()
    {
        if ( ! isLoaded)
        {
            String decoder = "decoder" + (decoderId + 1) + "/";
            int hiddenSize = settings.getHiddenSize();

            if (settings.isQueryKeyValueMerged())
            {
                // If the query, key and value matrices stored in the same file we have to split them
                float[][] qkvWeights = reader.readWeights(decoder + "att.query.key.value.w", hiddenSize * 3, hiddenSize);
                float[][][] qkvWeightSplit = Util.splitMatrix(qkvWeights, 3);

                float[] qkvBiases = reader.readVector(decoder + "att.query.key.value.b", hiddenSize * 3);
                float[][] qkvBiasSplit = Util.splitVector(qkvBiases, 3);

                this.queryWeights = qkvWeightSplit[0];
                this.queryBiases = qkvBiasSplit[0];
                this.keyWeights = qkvWeightSplit[1];
                this.keyBiases = qkvBiasSplit[1];
                this.valueWeights = qkvWeightSplit[2];
                this.valueBiases = qkvBiasSplit[2];
            }
            else
            {
                // Read the query, key and value matrices from separate files
                this.queryWeights = reader.readWeights(decoder + "att.query.w", hiddenSize, hiddenSize);
                this.queryBiases = reader.readVector(decoder + "att.query.b", hiddenSize);
                this.keyWeights = reader.readWeights(decoder + "att.key.w", hiddenSize, hiddenSize);
                this.keyBiases = reader.readVector(decoder + "att.key.b", hiddenSize);
                this.valueWeights = reader.readWeights(decoder + "att.value.w", hiddenSize, hiddenSize);
                this.valueBiases = reader.readVector(decoder + "att.value.b", hiddenSize);
            }

            this.projectionWeights = reader.readWeights(decoder +"att.proj.w",hiddenSize,hiddenSize);
            this.projectionBiases = reader.readVector(decoder +"att.proj.b",hiddenSize);
            this.attNormWeights = reader.readVector(decoder +"att.norm.w",hiddenSize);
            this.attNormBiases = reader.readVector(decoder +"att.norm.b",hiddenSize);
            this.mlpLayer1Weights = reader.readWeights(decoder +"mlp.layer1.w",hiddenSize *4,hiddenSize);
            this.mlpLayer1Biases = reader.readVector(decoder +"mlp.layer1.b",hiddenSize *4);
            this.mlpLayer2Weights = reader.readWeights(decoder +"mlp.layer2.w",hiddenSize,hiddenSize *4);
            this.mlpLayer2Biases = reader.readVector(decoder +"mlp.layer2.b",hiddenSize);
            this.mlpNormWeights = reader.readVector(decoder +"mlp.norm.w",hiddenSize);
            this.mlpNormBiases = reader.readVector(decoder +"mlp.norm.b",hiddenSize);

            isLoaded = true;
        }
    }

    public void clean()
    {
        if (settings.isCleanDecoder(decoderId))
        {
            isLoaded = false;

            this.queryWeights = null;
            this.queryBiases = null;
            this.keyWeights = null;
            this.keyBiases = null;
            this.valueWeights = null;
            this.valueBiases = null;
            this.projectionWeights = null;
            this.projectionBiases = null;
            this.attNormWeights = null;
            this.attNormBiases = null;
            this.mlpLayer1Weights = null;
            this.mlpLayer1Biases = null;
            this.mlpLayer2Weights = null;
            this.mlpLayer2Biases = null;
            this.mlpNormWeights = null;
            this.mlpNormBiases = null;
        }
    }

    public float[][] getQueryWeights()
    {
        init();
        return queryWeights;
    }

    public float[] getQueryBiases()
    {
        init();
        return queryBiases;
    }

    public float[][] getKeyWeights()
    {
        init();
        return keyWeights;
    }

    public float[] getKeyBiases()
    {
        init();
        return keyBiases;
    }

    public float[][] getValueWeights()
    {
        init();
        return valueWeights;
    }

    public float[] getValueBiases()
    {
        init();
        return valueBiases;
    }

    public float[][] getProjectionWeights()
    {
        init();
        return projectionWeights;
    }

    public float[] getProjectionBiases()
    {
        init();
        return projectionBiases;
    }

    public float[] getAttNormWeights()
    {
        init();
        return attNormWeights;
    }

    public float[] getAttNormBiases()
    {
        init();
        return attNormBiases;
    }

    public float[][] getMlpLayer1Weights()
    {
        init();
        return mlpLayer1Weights;
    }

    public float[] getMlpLayer1Biases()
    {
        init();
        return mlpLayer1Biases;
    }

    public float[][] getMlpLayer2Weights()
    {
        init();
        return mlpLayer2Weights;
    }

    public float[] getMlpLayer2Biases()
    {
        init();
        return mlpLayer2Biases;
    }

    public float[] getMlpNormWeights()
    {
        init();
        return mlpNormWeights;
    }

    public float[] getMlpNormBiases()
    {
        init();
        return mlpNormBiases;
    }
}
