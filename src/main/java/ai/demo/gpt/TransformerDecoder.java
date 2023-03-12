package ai.demo.gpt;

import java.util.ArrayList;
import java.util.List;
import static ai.demo.gpt.ParameterReader.*;
import static ai.demo.gpt.Settings.*;
import static ai.demo.gpt.TransformerUtil.*;

/**
 * Decoder implementation for a decoder-only transformer
 */
public class TransformerDecoder
{
    private final Settings settings;
    private final boolean hasAttention;
    private final int maxAttentionSize;
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
    private final List<float[][]> storedKeys = new ArrayList<>();
    private final List<float[][]> storedValues = new ArrayList<>();

    /**
     * Initialization
     */
    public TransformerDecoder(int decoderId, Settings settings, String attentionType)
    {
        this.settings = settings;
        this.hasAttention = !attentionType.equals(ATTENTION_NONE);
        this.maxAttentionSize = attentionType.equals(ATTENTION_LOCAL) ? settings.getLocalAttentionSize() : Integer.MAX_VALUE;

        String path = settings.getPath() + "/decoder" + (decoderId + 1);
        int hiddenSize = settings.getHiddenSize();

        this.queryWeights = readMatrixFile(path, "att.query.w", hiddenSize, hiddenSize);
        this.queryBiases = readVectorFile(path, "att.query.b", hiddenSize, settings.hasAttentionQueryBias());
        this.keyWeights = readMatrixFile(path, "att.key.w", hiddenSize, hiddenSize);
        this.keyBiases = readVectorFile(path, "att.key.b", hiddenSize, settings.hasAttentionKeyBias());
        this.valueWeights = readMatrixFile(path, "att.value.w", hiddenSize, hiddenSize);
        this.valueBiases = readVectorFile(path, "att.value.b", hiddenSize, settings.hasAttentionValueBias());
        this.projectionWeights = readMatrixFile(path, "att.proj.w", hiddenSize, hiddenSize);
        this.projectionBiases = readVectorFile(path, "att.proj.b", hiddenSize, settings.hasAttentionProjectionBias());
        this.attNormWeights = readVectorFile(path, "att.norm.w", hiddenSize);
        this.attNormBiases = readVectorFile(path, "att.norm.b", hiddenSize);
        this.mlpLayer1Weights = readMatrixFile(path, "mlp.layer1.w", hiddenSize, hiddenSize * 4);
        this.mlpLayer1Biases = readVectorFile(path, "mlp.layer1.b", hiddenSize * 4, settings.hasMlpLayer1Bias());
        this.mlpLayer2Weights = readMatrixFile(path, "mlp.layer2.w", hiddenSize * 4, hiddenSize);
        this.mlpLayer2Biases = readVectorFile(path, "mlp.layer2.b", hiddenSize, settings.hasMlpLayer2Bias());
        this.mlpNormWeights = readVectorFile(path, "mlp.norm.w", hiddenSize);
        this.mlpNormBiases = readVectorFile(path, "mlp.norm.b", hiddenSize);
    }

    /**
     * Decoder logic
     */
    public float[] execute(float[] hiddenState)
    {
        // Attention block
        if (hasAttention) hiddenState = attentionBlock(hiddenState);

        // Neuron layers
        return neuronBlock(hiddenState);
    }

    private float[] attentionBlock(float[] inputHiddenState)
    {
        // Normalization
        float[] hiddenState = normalization(inputHiddenState, attNormWeights, attNormBiases, settings.getEpsilon());

        // Attention layer
        hiddenState = attention(hiddenState);

        // Add the original input state to the actual (residual connection)
        return Util.addVectors(hiddenState, inputHiddenState);
    }

    private float[] neuronBlock(float[] inputHiddenState)
    {
        // Normalization
        float[] hiddenState = normalization(inputHiddenState, mlpNormWeights, mlpNormBiases, settings.getEpsilon());

        // Neuron layers
        hiddenState = neuronLayers(hiddenState);

        // Add the original input state to the actual (residual connection)
        return Util.addVectors(hiddenState, inputHiddenState);
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
            // Calculate the scores
            float[] scores = new float[storedKeys.size()];
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = Util.dotProduct(queries[head], storedKeys.get(pos)[head]) / settings.getScoreDividend();
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

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();
    }
}
