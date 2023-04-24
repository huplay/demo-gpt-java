package ai.demo.gpt.config;

import ai.demo.util.Util;
import ai.demo.util.Utility;

public class QueryKeyValueParameters
{
    private final int decoderId;

    private float[][] queryWeights;
    private float[] queryBiases;
    private float[][] keyWeights;
    private float[] keyBiases;
    private float[][] valueWeights;
    private float[] valueBiases;

    public QueryKeyValueParameters(int decoderId, Settings settings, ParameterReader reader)
    {
        this.decoderId = decoderId;

        Utility util = new Util();
        int hiddenSize = settings.getHiddenSize();

        if ( ! settings.isMergedQKV())
        {
            this.queryWeights = reader.readWeights(prefixedName("att.query.w"), hiddenSize, hiddenSize);
            this.queryBiases = reader.readVector(prefixedName("att.query.b"), hiddenSize);
            this.keyWeights = reader.readWeights(prefixedName("att.key.w"), hiddenSize, hiddenSize);
            this.keyBiases = reader.readVector(prefixedName("att.key.b"), hiddenSize);
            this.valueWeights = reader.readWeights(prefixedName("att.value.w"), hiddenSize, hiddenSize);
            this.valueBiases = reader.readVector(prefixedName("att.value.b"), hiddenSize);
        }
        else
        {
            int headCount = settings.getHeadCount();
            int headSize = settings.getHeadSize();

            float[] weights = reader.readVector(prefixedName("att.query.key.value.w"), 3 * hiddenSize * hiddenSize);
            float[] biases = reader.readVector(prefixedName("att.query.key.value.b"), 3 * hiddenSize);

            if (settings.isMergedQKVSplitHeadFirst())
            {
                // Weights
                float[][] weightsByHead = util.splitVector(weights, settings.getHeadCount());

                float[][] queryWeightsByHead = new float[headCount][headSize * headSize];
                float[][] keyWeightsByHead = new float[headCount][headSize * headSize];
                float[][] valueWeightsByHead = new float[headCount][headSize * headSize];

                for (int i = 0; i < headCount; i++)
                {
                    float[][] split = util.splitVector(weightsByHead[i], 3);
                    queryWeightsByHead[i] = split[0];
                    keyWeightsByHead[i] = split[1];
                    valueWeightsByHead[i] = split[2];
                }

                this.queryWeights = util.splitVector(util.flattenMatrix(queryWeightsByHead), hiddenSize);
                this.keyWeights = util.splitVector(util.flattenMatrix(keyWeightsByHead), hiddenSize);
                this.valueWeights = util.splitVector(util.flattenMatrix(valueWeightsByHead), hiddenSize);

                // Biases
                float[][] biasesByHead = util.splitVector(biases, settings.getHeadCount());

                float[][] queryBiasesByHead = new float[headCount][headSize];
                float[][] keyBiasesByHead = new float[headCount][headSize];
                float[][] valueBiasesByHead = new float[headCount][headSize];

                for (int i = 0; i < headCount; i++)
                {
                    float[][] split = util.splitVector(biasesByHead[i], 3);
                    queryBiasesByHead[i] = split[0];
                    keyBiasesByHead[i] = split[1];
                    valueBiasesByHead[i] = split[2];
                }

                this.queryBiases = util.flattenMatrix(queryBiasesByHead);
                this.keyBiases = util.flattenMatrix(keyBiasesByHead);
                this.valueBiases = util.flattenMatrix(valueBiasesByHead);
            }
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
}
