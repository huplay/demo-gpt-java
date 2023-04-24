package ai.demo.gpt;

import static ai.demo.gpt.App.UTIL;
import static java.lang.Math.*;

public class NeuronUtil
{
    /**
     * Applying weights and biases
     */
    public static float[] applyParams(float[] vector, float[][] weights, float[] biases)
    {
        // Apply weights
        float[] result = UTIL.multiplyVectorByTransposedMatrix(vector, weights);

        // Apply biases (optional)
        if (biases != null)
        {
            result = UTIL.addVectors(result, biases);
        }

        return result;
    }

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    public static float gelu(float value)
    {
        // Using a constant for sqrt(2 / PI) didn't make it faster, most likely Java optimized it
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * value * value * value))));
    }
}
