package ai.demo.gpt;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;
import static java.lang.Math.exp;
import static java.lang.Math.sqrt;

public class Util
{
    /**
     * Vector to vector addition
     */
    public static float[] addVectors(float[] vector1, float[] vector2)
    {
        float[] ret = new float[vector1.length];

        for (int i = 0; i < vector1.length; i++)
        {
            ret[i] = vector1[i] + vector2[i];
        }

        return ret;
    }

    /**
     * Dot product calculation (multiplying vector by vector)
     */
    public static float dotProduct(float[] vector1, float[] vector2)
    {
        float sum = 0;

        for (int i = 0; i < vector1.length; i++)
        {
            sum = sum + vector1[i] * vector2[i];
        }

        return sum;
    }

    /**
     * Multiply vector by a scalar
     */
    public static float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            ret[i] = vector[i] * scalar;
        }

        return ret;
    }

    /**
     * Multiply vector by matrix
     */
    public static float[] multiplyVectorByMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix[0].length];

        for (int col = 0; col < matrix[0].length; col++)
        {
            float sum = 0;

            for (int i = 0; i < vector.length; i++)
            {
                sum = sum + vector[i] * matrix[i][col];
            }

            ret[col] = sum;
        }

        return ret;
    }

    /**
     * Multiply vector by transposed matrix
     */
    public static float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix.length];

        for (int row = 0; row < matrix.length; row++)
        {
            float sum = 0;

            for (int i = 0; i < vector.length; i++)
            {
                sum = sum + vector[i] * matrix[row][i];
            }

            ret[row] = sum;
        }

        return ret;
    }

    /**
     * Split a vector to a matrix
     */
    public static float[][] splitVector(float[] vector, int count)
    {
        int size = vector.length / count;
        float[][] ret = new float[count][size];

        int segment = 0;
        int col = 0;
        for (float value : vector)
        {
            ret[segment][col] = value;

            if (col == size - 1)
            {
                col = 0;
                segment++;
            }
            else col++;
        }

        return ret;
    }

    /**
     * Merge the rows of a matrix to a single vector
     */
    public static float[] flattenMatrix(float[][] matrix)
    {
        float[] ret = new float[matrix.length * matrix[0].length];

        int i = 0;

        for (float[] row : matrix)
        {
            for (float value : row)
            {
                ret[i] = value;
                i++;
            }
        }

        return ret;
    }

    /**
     * Calculate average (mean) value
     */
    public static float average(float[] vector)
    {
        double sum = 0;

        for (float value : vector)
        {
            sum = sum + value;
        }

        return (float) sum / vector.length;
    }

    /**
     * Calculate the average difference -
     */
    public static float averageDiff(float[] values, float average, float epsilon)
    {
        float[] squareDiff = new float[values.length];

        for (int i = 0; i < values.length; i++)
        {
            float diff = values[i] - average;
            squareDiff[i] = diff * diff;
        }

        float averageSquareDiff = average(squareDiff);

        return (float) sqrt(averageSquareDiff + epsilon);
    }

    /**
     * Standard normalization - (value - avg) * sqrt( (value - avg)^2 + epsilon)
     */
    public static float[] normalize(float[] vector, float epsilon)
    {
        float average = average(vector);
        float averageDiff = averageDiff(vector, average, epsilon);

        float[] norm = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            norm[i] = (vector[i] - average) / averageDiff;
        }

        return norm;
    }

    /**
     * Calculate softmax - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(float[] vector)
    {
        double total = 0;
        for (float value : vector)
        {
            total = total + exp(value);
        }

        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            ret[i] = (float) (exp(vector[i]) / total);
        }

        return ret;
    }

    /**
     * Calculate softmax on IndexedValue list - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(List<IndexedValue> values)
    {
        double total = 0;
        for (IndexedValue value : values)
        {
            total = total + exp(value.value);
        }

        float[] ret = new float[values.size()];

        for (int i = 0; i < values.size(); i++)
        {
            ret[i] = (float) (exp(values.get(i).value) / total);
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

    /**
     * Sort values to reversed order and filter out the lowest values (retain the top [count] values)
     */
    public static List<IndexedValue> reverseAndFilter(float[] values, int count)
    {
        TreeSet<IndexedValue> indexedValues = new TreeSet<>(new ReverseComparator());
        for (int i = 0; i < values.length; i++)
        {
            indexedValues.add(new IndexedValue(values[i], i));
        }

        List<IndexedValue> filteredValues = new ArrayList<>(count);

        int i = 0;
        for (IndexedValue indexedValue : indexedValues)
        {
            filteredValues.add(indexedValue);
            i++;
            if (i == count) break;
        }

        return filteredValues;
    }

    /**
     * Holder of a value with the index (position of the element)
     */
    public static class IndexedValue
    {
        public float value;
        public int index;

        public IndexedValue(float value, int index)
        {
            this.value = value;
            this.index = index;
        }
    }

    /**
     * Comparator for IndexedValue to achieve reverse ordering
     */
    private static class ReverseComparator implements Comparator<IndexedValue>
    {
        public int compare(IndexedValue a, IndexedValue b)
        {
            return Float.compare(b.value, a.value);
        }
    }
}
