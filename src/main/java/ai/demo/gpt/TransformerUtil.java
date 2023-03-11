package ai.demo.gpt;

import static java.lang.Math.*;

public class TransformerUtil
{
    /**
     * Applying weights using vector by matrix multiplication plus adding biases
     */
    public static float[] applyWeight(float[] vector, float[][] weights, float[] biases)
    {
        float[] result = Util.multiplyVectorByMatrix(vector, weights);

        if (biases != null)
        {
            result = Util.addVectors(result, biases);
        }

        return result;
    }

    /**
     * Standard normalization with applying weights and biases
     */
    public static float[] normalization(float[] vector, float[] weights, float[] biases, float epsilon)
    {
        // Standard normalization
        float[] result = Util.normalize(vector, epsilon);

        // Applying the trained weights and biases
        for (int i = 0; i < vector.length; i++)
        {
            result[i] = result[i] * weights[i] + biases[i];
        }

        return result;
    }

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: https://paperswithcode.com/method/gelu
     */
    public static float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * value * value * value))));
    }
}
