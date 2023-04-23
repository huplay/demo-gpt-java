package ai.demo.gpt;

import ai.demo.gpt.config.DecoderParameters;
import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;
import ai.demo.gpt.position.PositionEmbedder;
import ai.demo.util.Util;

import java.util.ArrayList;
import java.util.List;
import static ai.demo.gpt.TransformerUtil.*;

/**
 * Decoder implementation for a decoder-only transformer
 */
public class TransformerDecoder
{
    private final int decoderId;
    private final boolean lastDecoder;
    private final Settings settings;
    private final PositionEmbedder position;
    private final DecoderParameters params;
    private final NeuronUtil neuronUtil;
    private final float epsilon;

    private final List<float[][]> storedKeys = new ArrayList<>();
    private final List<float[][]> storedValues = new ArrayList<>();

    public TransformerDecoder(int decoderId, Settings settings, ParameterReader reader, PositionEmbedder position)
    {
        this.decoderId = decoderId;
        this.lastDecoder = (decoderId == settings.getDecoderCount());
        this.settings = settings;
        this.params = new DecoderParameters(decoderId, settings, reader);
        this.position = position;
        this.neuronUtil = new NeuronUtil(params);
        this.epsilon = settings.getEpsilon();
    }

    public float[] execute(float[] hiddenState, boolean withOutput)
    {
        // Attention block
        if (settings.hasAttention(decoderId))
        {
            hiddenState = attentionBlock(hiddenState);
        }

        // Neuron layers
        if ( withOutput || ! lastDecoder)
        {
            hiddenState = neuronBlock(hiddenState);
        }

        return hiddenState;
    }

    private float[] attentionBlock(float[] inputHiddenState)
    {
        if (settings.isPreNormalization())
        {
            // These are the steps for most of the models:
            float[] hiddenState = norm(inputHiddenState, params.getAttNormWeights(), params.getAttNormBiases(), epsilon);

            hiddenState = attention(hiddenState);

            return Util.addVectors(inputHiddenState, hiddenState);
        }
        else
        {
            // GPT-1 and similarly old models used these steps:
            float[] hiddenState = attention(inputHiddenState);

            hiddenState = Util.addVectors(inputHiddenState, hiddenState);

            return norm(hiddenState, params.getAttNormWeights(), params.getAttNormBiases(), epsilon);
        }
    }

    private float[] neuronBlock(float[] inputHiddenState)
    {
        if (settings.isPreNormalization())
        {
            // These are the steps for most of the models:
            float[] hiddenState = norm(inputHiddenState, params.getMlpNormWeights(), params.getMlpNormBiases(), epsilon);

            hiddenState = neuronLayers(hiddenState);

            return Util.addVectors(inputHiddenState, hiddenState);
        }
        else
        {
            // GPT-1 and similarly old models used these steps:
            float[] hiddenState = neuronLayers(inputHiddenState);

            hiddenState = Util.addVectors(inputHiddenState, hiddenState);

            return norm(hiddenState, params.getMlpNormWeights(), params.getMlpNormBiases(), epsilon);
        }
    }

    private float[] attention(float[] hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token:
        float[] query = neuronUtil.applyParams(hiddenState, params.getQueryWeights(), params.getQueryBiases(), "att.query.w");
        float[] key = neuronUtil.applyParams(hiddenState, params.getKeyWeights(), params.getKeyBiases(), "att.key.w");
        float[] value = neuronUtil.applyParams(hiddenState, params.getValueWeights(), params.getValueBiases(), "att.value.w");

        // Split the query, key and value vectors into pieces for all heads
        float[][] queryByHead = Util.splitVector(query, settings.getHeadCount());
        float[][] keyByHead = Util.splitVector(key, settings.getHeadCount());
        float[][] valueByHead = Util.splitVector(value, settings.getHeadCount());

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

        // Used only at sparse attention:
        if (storedSize > settings.getMaxAttentionSize(decoderId))
        {
            // Topping the maximum attention size we can drop the oldest stored values
            storedKeys.remove(0);
            storedValues.remove(0);
        }

        // Declaration of the variable for collecting the attention results for all heads
        float[][] valueAggregate = new float[settings.getHeadCount()][settings.getHeadSize()];

        // Scoring the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < settings.getHeadCount(); head++)
        {
            float[] actualQuery = queryByHead[head]; /* Optional: */ actualQuery = position.toQuery(actualQuery, storedSize, storedSize - 1, head);

            // Calculate the scores
            float[] scores = new float[storedSize];

            for (int pos = 0; pos < storedSize; pos++)
            {
                float[] relatedKey = storedKeys.get(pos)[head]; /* Optional: */ relatedKey = position.toKey(relatedKey, storedSize, pos, head);

                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                float score = Util.dotProduct(actualQuery, relatedKey); /* Optional: */ score = position.toScore(score, storedSize, pos, head);

                scores[pos] = score / settings.getAttentionDividend();
            }

            // Rescaling the scores to values between 0 and 1
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                float[] relatedValue = storedValues.get(pos)[head]; /* Optional: */ relatedValue = position.toValue(relatedValue, storedSize, pos, head);

                float[] multipliedValue = Util.multiplyVectorByScalar(relatedValue, scores[pos]);
                
                valueAggregate[head] = Util.addVectors(valueAggregate[head], multipliedValue);
            }
        }

        // Concatenate the results for all heads
        float[] flatSums = Util.flattenMatrix(valueAggregate);

        // Apply the attention projection weights and biases
        return neuronUtil.applyParams(flatSums, params.getProjectionWeights(), params.getProjectionBiases(), "att.proj.w");
    }

    private float[] neuronLayers(float[] hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = neuronUtil.applyParams(hiddenState, params.getMlpLayer1Weights(), params.getMlpLayer1Biases(), "mlp.layer1.w");

        for (int neuron = 0; neuron < settings.getFeedForwardSize(); neuron++)
        {
            hiddenState[neuron] = neuronUtil.gelu(hiddenState[neuron]);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        return neuronUtil.applyParams(hiddenState, params.getMlpLayer2Weights(), params.getMlpLayer2Biases(), "mlp.layer2.w");
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
