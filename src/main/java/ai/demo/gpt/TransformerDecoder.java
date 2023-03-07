package ai.demo.gpt;

import java.util.ArrayList;
import java.util.List;
import static ai.demo.gpt.ParameterReader.*;
import static ai.demo.gpt.Settings.*;
import static java.lang.Math.*;
import static java.lang.Math.pow;

/**
 * Decoder implementation for a decoder-only transformer
 */
public class TransformerDecoder
{
    private final Settings settings;
    private final boolean isAttention;
    private final int maxAttentionSize;
    private final float[][] queryWeighs;
    private final float[] queryBiases;
    private final float[][] keyWeighs;
    private final float[] keyBiases;
    private final float[][] valueWeighs;
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
        String path = settings.getPath() + "/decoder" + (decoderId + 1) + "/";
        int size = settings.getEmbeddingSize();

        this.settings = settings;
        this.isAttention = !attentionType.equals(ATTENTION_NONE);
        this.maxAttentionSize = attentionType.equals(ATTENTION_LOCAL) ? settings.getLocalAttentionSize() : Integer.MAX_VALUE;
        this.queryWeighs = readMatrixFile(path + ATT_QUERY_W_DAT, size, size);
        this.queryBiases = readVectorFile(path + ATT_QUERY_B_DAT, size);
        this.keyWeighs = readMatrixFile(path + ATT_KEY_W_DAT, size, size);
        this.keyBiases = readVectorFile(path + ATT_KEY_B_DAT, size);
        this.valueWeighs = readMatrixFile(path + ATT_VALUE_W_DAT, size, size);
        this.valueBiases = readVectorFile(path + ATT_VALUE_B_DAT, size);
        this.projectionWeights = readMatrixFile(path + ATT_PROJ_W_DAT, size, size);
        this.projectionBiases = readVectorFile(path + ATT_PROJ_B_DAT, size);
        this.attNormWeights = readVectorFile(path + ATT_NORM_W_DAT, size);
        this.attNormBiases = readVectorFile(path + ATT_NORM_B_DAT, size);
        this.mlpLayer1Weights = readMatrixFile(path + MLP_LAYER1_W_DAT, size, size * 4);
        this.mlpLayer1Biases = readVectorFile(path + MLP_LAYER1_B_DAT, size * 4);
        this.mlpLayer2Weights = readMatrixFile(path + MLP_LAYER2_W_DAT, size * 4, size);
        this.mlpLayer2Biases = readVectorFile(path + MLP_LAYER2_B_DAT, size);
        this.mlpNormWeights = readVectorFile(path + MLP_NORM_W_DAT, size);
        this.mlpNormBiases = readVectorFile(path + MLP_NORM_B_DAT, size);
    }

    /**
     * Decoder logic
     */
    public float[] execute(float[] hiddenState)
    {
        // Attention block
        if (isAttention) hiddenState = attentionBlock(hiddenState);

        // Feedforward block
        return mlpBlock(hiddenState);
    }

    private float[] attentionBlock(float[] inputHiddenState)
    {
        // Normalization
        float[] hiddenState = normalization(inputHiddenState, attNormWeights, attNormBiases);

        // Attention layer
        hiddenState = attention(hiddenState);

        // Add the original input state to the actual (residual connection)
        return Util.addVectors(hiddenState, inputHiddenState);
    }

    private float[] mlpBlock(float[] inputHiddenState)
    {
        // Normalization
        float[] hiddenState = normalization(inputHiddenState, mlpNormWeights, mlpNormBiases);

        // Feedforward layers
        hiddenState = mlp(hiddenState);

        // Add the original input state to the actual (residual connection)
        return Util.addVectors(hiddenState, inputHiddenState);
    }

    private float[] attention(float[] hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token:
        float[] query = Util.multiplyVectorByMatrix(hiddenState, queryWeighs);
        query = Util.addVectors(query, queryBiases);

        float[] key = Util.multiplyVectorByMatrix(hiddenState, keyWeighs);
        key = Util.addVectors(key, keyBiases);

        float[] value = Util.multiplyVectorByMatrix(hiddenState, valueWeighs);
        value = Util.addVectors(value, valueBiases);

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

        float[][] sums = new float[settings.getHeadCount()][settings.getEmbeddingSize() / settings.getHeadCount()];

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
            scores = Util.softmax(scores);

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
        hiddenState = Util.multiplyVectorByMatrix(flatSums, projectionWeights);
        return Util.addVectors(hiddenState, projectionBiases);
    }

    private float[] mlp(float[] hiddenState)
    {
        // Two layered feedforward neural network:
        // - Layer 1: <embeddingSize> * 4 neurons (using a gelu activation function)
        // - Layer 2: <embeddingSize> neurons (without activation function, simply resulting the weighted + biased input)

        // The calculation of a neuron layer is simply a multiplication of the input vector by the weight matrix,
        // and a vector to vector addition using the biases, finally executing the activation function

        // Layer 1:
        hiddenState = Util.multiplyVectorByMatrix(hiddenState, mlpLayer1Weights);
        hiddenState = Util.addVectors(hiddenState, mlpLayer1Biases);

        // Using the gelu activation function, calculating the output of the first layer
        for (int neuron = 0; neuron < 4 * settings.getEmbeddingSize(); neuron++)
        {
            hiddenState[neuron] = gelu(hiddenState[neuron]);
        }

        // Layer 2:
        hiddenState = Util.multiplyVectorByMatrix(hiddenState, mlpLayer2Weights);
        return Util.addVectors(hiddenState, mlpLayer2Biases);
    }

    // Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
    private static float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * pow(value, 3)))));
    }

    private float[] normalization(float[] hiddenState, float[] weights, float[] biases)
    {
        // Standard normalization
        hiddenState = Util.normalize(hiddenState, settings.getEpsilon());

        // Applying the trained weights and biases
        for (int i = 0; i < hiddenState.length; i++)
        {
            hiddenState[i] = hiddenState[i] * weights[i] + biases[i];
        }

        return hiddenState;
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
