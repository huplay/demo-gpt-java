package ai.demo.gpt;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;
import ai.demo.gpt.position.PositionEmbedder;
import ai.demo.util.Util;

import java.util.ArrayList;
import java.util.List;
import static ai.demo.gpt.config.Settings.*;
import static ai.demo.gpt.TransformerUtil.*;

/**
 * Decoder implementation for a decoder-only transformer
 */
public class TransformerDecoder
{
    private final int decoderId;
    private final Settings settings;
    private final ParameterReader reader;
    private final PositionEmbedder positionEmbedder;
    private final boolean isPreNormalization;
    private final float epsilon;
    private final boolean hasAttention;
    private final int maxAttentionSize;

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

    private final List<float[][]> storedKeys = new ArrayList<>();
    private final List<float[][]> storedValues = new ArrayList<>();

    private boolean isLoaded = false;

    /**
     * Initialization
     */
    public TransformerDecoder(int decoderId, Settings settings, ParameterReader reader, PositionEmbedder positionEmbedder)
    {
        this.decoderId = decoderId;
        this.settings = settings;
        this.reader = reader;
        this.positionEmbedder = positionEmbedder;
        this.isPreNormalization = settings.isPreNormalization();
        this.epsilon =settings.getEpsilon();

        String attentionType = settings.getAttentionType()[decoderId];
        this.hasAttention = !attentionType.equals(ATTENTION_NONE);
        this.maxAttentionSize = attentionType.equals(ATTENTION_LOCAL)?settings.getLocalAttentionSize():Integer.MAX_VALUE;
    }

    public float[] execute(float[] hiddenState)
    {
        init();

        // Attention block
        if (hasAttention)
        {
            hiddenState = attentionBlock(hiddenState);
        }

        // Neuron layers
        hiddenState = neuronBlock(hiddenState);

        clean();

        return hiddenState;
    }

    private float[] attentionBlock(float[] inputHiddenState)
    {
        if (settings.isPreNormalization())
        {
            float[] hiddenState = normalization(inputHiddenState, attNormWeights, attNormBiases, epsilon);

            hiddenState = attention(hiddenState);

            return Util.addVectors(inputHiddenState, hiddenState);
        }
        else
        {
            float[] hiddenState = attention(inputHiddenState);

            hiddenState = Util.addVectors(inputHiddenState, hiddenState);

            return normalization(hiddenState, attNormWeights, attNormBiases, epsilon);
        }
    }

    private float[] neuronBlock(float[] inputHiddenState)
    {
        if (settings.isPreNormalization())
        {
            float[] hiddenState = normalization(inputHiddenState, mlpNormWeights, mlpNormBiases, epsilon);

            hiddenState = neuronLayers(hiddenState);

            return Util.addVectors(inputHiddenState, hiddenState);
        }
        else
        {
            float[] hiddenState = neuronLayers(inputHiddenState);

            hiddenState = Util.addVectors(inputHiddenState, hiddenState);

            return normalization(hiddenState, mlpNormWeights, mlpNormBiases, epsilon);
        }
    }

    private float[] attention(float[] hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token:
        float[] query = applyWeight(hiddenState, queryWeights, queryBiases);
        float[] key = applyWeight(hiddenState, keyWeights, keyBiases);
        float[] value = applyWeight(hiddenState, valueWeights, valueBiases);

        // Split the query, key and value vectors into pieces for all heads
        float[][] queries = Util.splitVector(query, settings.getHeadCount());
        float[][] keys = Util.splitVector(key, settings.getHeadCount());
        float[][] values = Util.splitVector(value, settings.getHeadCount());

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keys);
        storedValues.add(values);

        // Using local attention there is a maximum attention size (otherwise the max in practice infinite)
        if (storedKeys.size() > maxAttentionSize)
        {
            // Topping the maximum attention size we can drop the oldest stored values
            storedKeys.remove(0);
            storedValues.remove(0);
        }

        float[][] sums = new float[settings.getHeadCount()][settings.getHiddenSize() / settings.getHeadCount()];

        // Scoring the previous tokens (including the actual), separately for all heads
        // Again: we have to score not only the previous, but the actual token as well
        // That is the reason of that we already added the actual key/value to the stored keys/values
        for (int head = 0; head < settings.getHeadCount(); head++)
        {
            float[] q = positionEmbedder.addRelativePosition(queries[head], storedKeys.size() - 1);

            // Calculate the scores
            float[] scores = new float[storedKeys.size()];
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                float[] k = positionEmbedder.addRelativePosition(storedKeys.get(pos)[head], pos);

                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = Util.dotProduct(q, k) / settings.getAttentionDividend();
            }

            // Softmax
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                float[] sum = Util.multiplyVectorByScalar(storedValues.get(pos)[head], scores[pos]);
                sums[head] = Util.addVectors(sums[head], sum);
            }
        }

        // Concatenate the results for all heads
        float[] flatSums = Util.flattenMatrix(sums);

        // Apply the attention projection weights and biases
        return applyWeight(flatSums, projectionWeights, projectionBiases);
    }

    private float[] neuronLayers(float[] hiddenState)
    {
        // Layer 1: <hiddenSize> * 4 neurons (using a gelu activation function)
        hiddenState = applyWeight(hiddenState, mlpLayer1Weights, mlpLayer1Biases);
        for (int neuron = 0; neuron < settings.getHiddenSize() * 4; neuron++)
        {
            hiddenState[neuron] = gelu(hiddenState[neuron]);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        return applyWeight(hiddenState, mlpLayer2Weights, mlpLayer2Biases);
    }

    private void init()
    {
        if ( ! isLoaded)
        {
            String decoder = "decoder" + (decoderId + 1) + "/";
            int hiddenSize = settings.getHiddenSize();

            if(settings.isQueryKeyValueMerged())
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

    private void clean()
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

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();
    }
}
