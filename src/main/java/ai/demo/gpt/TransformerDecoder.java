package ai.demo.gpt;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.*;
import static java.lang.Math.pow;

/**
 * Decoder implementation for a decoder-only Transformer
 */
public class TransformerDecoder
{
    private final TrainedParameters.DecoderParameters params;

    private final boolean isSparse;
    private final float epsilon;
    private final int embeddingSize;
    private final int headCount;
    private final int attentionDividend;

    private final List<float[][]> storedKeys = new ArrayList<>();
    private final List<float[][]> storedValues = new ArrayList<>();

    /**
     * Initialization
     */
    public TransformerDecoder(Config config, boolean isSparse, TrainedParameters.DecoderParameters params, float epsilon)
    {
        this.params = params;
        this.isSparse = isSparse;
        this.epsilon = epsilon;
        this.embeddingSize = config.getModelType().embeddingSize;
        this.headCount = config.getModelType().headCount;

        // The vector size is always 64, so this is always 8, it is possible to convert to int.
        this.attentionDividend = (int) sqrt((float) embeddingSize / headCount);
    }

    /**
     * Decoder logic
     */
    public float[] execute(float[] hiddenState)
    {
        // Attention block
        hiddenState = attentionBlock(hiddenState);

        // Feed forward block
        return feedForwardBlock(hiddenState);
    }

    private float[] attentionBlock(float[] inputHiddenState)
    {
        // Normalization
        float[] hiddenState = normalize(inputHiddenState, params.norm1Weights, params.norm1Biases);

        // Attention layer
        hiddenState = attention(hiddenState);

        // Residual connection
        return Util.addVectors(hiddenState, inputHiddenState);
    }

    private float[] feedForwardBlock(float[] inputHiddenState)
    {
        // Normalization
        float[] hiddenState = normalize(inputHiddenState, params.norm2Weights, params.norm2Biases);

        // Feed forward layers
        hiddenState = feedForward(hiddenState);

        // Residual connection
        return Util.addVectors(hiddenState, inputHiddenState);
    }

    private float[] attention(float[] hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token:
        float[] query = Util.multiplyVectorByMatrix(hiddenState, params.queryWeighs);
        query = Util.addVectors(query, params.queryBiases);

        float[] key = Util.multiplyVectorByMatrix(hiddenState, params.keyWeighs);
        key = Util.addVectors(key, params.keyBiases);

        float[] value = Util.multiplyVectorByMatrix(hiddenState, params.valueWeighs);
        value = Util.addVectors(value, params.valueBiases);

        // Split the query, key and value vectors into pieces for all heads
        float[][] queries = Util.splitVector(query, headCount);
        float[][] keys = Util.splitVector(key, headCount);
        float[][] values = Util.splitVector(value, headCount);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keys);
        storedValues.add(values);

        float[][] sums = new float[headCount][embeddingSize / headCount];

        // Scoring the previous tokens (including the actual), separately for all heads
        // Again: we have to score not only the previous, but the actual token as well
        // That is the reason of that we already added the actual key/value to the stored keys/values
        for (int head = 0; head < headCount; head++)
        {
            // Calculate the scores
            float[] scores = new float[storedKeys.size()];
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = Util.dotProduct(queries[head], storedKeys.get(pos)[head]) / attentionDividend;
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
        hiddenState = Util.multiplyVectorByMatrix(flatSums, params.projectionWeights);
        return Util.addVectors(hiddenState, params.projectionBiases);
    }

    private float[] feedForward(float[] hiddenState)
    {
        // Two layered feed forward neural network:
        // - Layer 1: <embeddingSize> * 4 neurons (using a gelu activation function)
        // - Layer 2: <embeddingSize> neurons (without activation function, simply resulting the weighted + biased input)

        // The calculation of a neuron layer is simply a multiplication of the input vector by the weight matrix,
        // and a vector to vector addition using the biases, finally executing the activation function

        // Layer 1:
        hiddenState = Util.multiplyVectorByMatrix(hiddenState, params.feedForwardLayer1Weights);
        hiddenState = Util.addVectors(hiddenState, params.feedForwardLayer1Biases);

        // Using the gelu activation function, calculating the output of the first layer
        for (int neuron = 0; neuron < 4 * embeddingSize; neuron++)
        {
            hiddenState[neuron] = gelu(hiddenState[neuron]);
        }

        // Layer 2:
        hiddenState = Util.multiplyVectorByMatrix(hiddenState, params.feedForwardLayer2Weights);
        return Util.addVectors(hiddenState, params.feedForwardLayer2Biases);
    }

    // Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
    private static float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * pow(value, 3)))));
    }

    private float[] normalize(float[] hiddenState, float[] weights, float[] biases)
    {
        // Standard normalization
        hiddenState = Util.normalize(hiddenState, epsilon);

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
