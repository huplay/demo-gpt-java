package ai.demo.gpt;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static java.lang.Math.sqrt;

public class Util
{
    /**
     * Vector to vector addition
     */
    public static float[] addVectors(float[] vector1, float[] vector2)
    {
        try (INDArray array1 = Nd4j.create(vector1);
             INDArray array2 = Nd4j.create(vector2))
        {
            return array1.add(array2).toFloatVector();
        }
    }

    /**
     * Dot product calculation (multiplying vector by vector)
     */
    public static float dotProduct(float[] vector1, float[] vector2)
    {
        return Nd4j.create(vector1).mmul(Nd4j.create(vector2)).getFloat(0);
    }

    /**
     * Multiply vector by a scalar
     */
    public static float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        return Nd4j.create(vector).mul(scalar).toFloatVector();
    }

    /**
     * Multiply vector by matrix
     */
    public static float[] multiplyVectorByMatrix(float[] vector, float[][] matrix)
    {
        return Nd4j.create(new float[][] {vector}).mmul(Nd4j.create(matrix)).toFloatVector();
    }

    /**
     * Multiply vector by transposed matrix
     */
    public static float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[][] a2 = new float[1][vector.length];
        a2[0] = vector;

        return Nd4j.create(a2).mmul(Nd4j.create(matrix).transpose()).toFloatVector();
    }

    /**
     * Split a vector to a matrix
     */
    public static float[][] splitVector(float[] vector, int count)
    {
        return Nd4j.create(vector).reshape(count, vector.length / count).toFloatMatrix();
    }

    /**
     * Merge the rows of a matrix to a single vector
     */
    public static float[] flattenMatrix(float[][] matrix)
    {
        return Nd4j.create(matrix).reshape(matrix.length * matrix[0].length).toFloatVector();
    }

    /**
     * Calculate average (mean) value
     */
    public static float average(float[] vector)
    {
        return Nd4j.create(vector).meanNumber().floatValue();
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
}