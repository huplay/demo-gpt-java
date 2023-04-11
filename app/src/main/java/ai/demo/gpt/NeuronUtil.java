package ai.demo.gpt;

import ai.demo.gpt.config.DecoderParameters;
import ai.demo.util.Util;

import static java.lang.Math.*;

public class NeuronUtil
{
    private final DecoderParameters decoderParameters;

    public NeuronUtil(DecoderParameters decoderParameters)
    {
        this.decoderParameters = decoderParameters;
    }

    /**
     * Applying weights and biases
     */
    public float[] applyParams(float[] vector, float[][] weights, float[] biases, String name)
    {
        if (weights == null)
        {
            // If the weight is empty hopefully this is a memory saver decoder: read the weights on the fly
            weights = decoderParameters.readWeights(name, true);
        }

        // Apply weights
        float[] result = Util.multiplyVectorByMatrix(vector, weights);

        // Apply biases (optional)
        if (biases != null)
        {
            result = Util.addVectors(result, biases);
        }

        return result;
    }

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    public float gelu(float value)
    {
        // Using a constant for sqrt(2 / PI) didn't make it faster, most likely Java optimized it
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * value * value * value))));
    }
}
