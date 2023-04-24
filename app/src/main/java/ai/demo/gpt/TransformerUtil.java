package ai.demo.gpt;

import ai.demo.util.IndexedValue;
import java.util.List;

import static ai.demo.gpt.App.UTIL;
import static java.lang.Math.*;

public class TransformerUtil
{
    /**
     * Standard normalization with applying normalization weights and biases
     */
    public static float[] norm(float[] vector, float[] weights, float[] biases, float epsilon)
    {
        // Standard normalization
        float[] result = UTIL.normalize(vector, epsilon);

        // Applying the trained weights and biases
        for (int i = 0; i < vector.length; i++)
        {
            result[i] = result[i] * weights[i] + biases[i];
        }

        return result;
    }

    /**
     * Calculate softmax - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(float[] vector)
    {
        float max = UTIL.max(vector);

        double total = 0;
        for (float value : vector)
        {
            double exp = exp(value - max);

            total = total + exp;
        }

        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            double exp = exp(vector[i] - max);

            ret[i] = (float) (exp / total);
        }

        return ret;
    }

    /**
     * Calculate softmax on IndexedValue list - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(List<IndexedValue> values)
    {
        float max = UTIL.max(values);

        double total = 0;
        for (IndexedValue value : values)
        {
            total = total + exp(value.getValue() - max);
        }

        float[] ret = new float[values.size()];

        for (int i = 0; i < values.size(); i++)
        {
            ret[i] = (float) (exp(values.get(i).getValue() - max) / total);
        }

        return ret;
    }

    /**
     * Weighted random selection from list of probabilities
     */
    public static int weightedRandomPick(float[] probabilities)
    {
        float sum = 0;
        float[] cumulativeProbabilities = new float[probabilities.length];

        for (int i = 0; i < probabilities.length; i++)
        {
            sum = sum + probabilities[i] * 100;
            cumulativeProbabilities[i] = sum;
        }

        int random = (int)(Math.random() * sum);

        int index = 0;
        for (int i = 0; i < probabilities.length; i++)
        {
            if (random < cumulativeProbabilities[i]) break;

            index ++;
        }

        return index;
    }
}
