package ai.demo.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static java.lang.Math.sqrt;

public class Util
{
    public static String getUtilName()
    {
        return "ND4j";
    }

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
        try (INDArray array1 = Nd4j.create(vector1);
             INDArray array2 = Nd4j.create(vector2))
        {
            return array1.mmul(array2).getFloat(0);
        }
    }

    /**
     * Multiply vector by a scalar
     */
    public static float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        try (INDArray array = Nd4j.create(vector))
        {
            return array.mul(scalar).toFloatVector();
        }
    }

    /**
     * Multiply vector by matrix
     */
    public static float[] multiplyVectorByMatrix(float[] vector, float[][] matrix)
    {
        try (INDArray array1 = Nd4j.create(new float[][] {vector});
             INDArray array2 = Nd4j.create(matrix).transpose())
        {
            return array1.mmul(array2).toFloatVector();
        }
    }

    /**
     * Multiply vector by transposed matrix
     */
    public static float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[][] array = new float[1][vector.length];
        array[0] = vector;

        try (INDArray array1 = Nd4j.create(array);
             INDArray array2 = Nd4j.create(matrix))
        {
            return array1.mmul(array2.transpose()).toFloatVector();
        }
    }

    /**
     * Split a vector to a matrix
     */
    public static float[][] splitVector(float[] vector, int count)
    {
        try (INDArray array = Nd4j.create(vector))
        {
            return array.reshape(count, vector.length / count).toFloatMatrix();
        }
    }

    /**
     * Merge the rows of a matrix to a single vector
     */
    public static float[] flattenMatrix(float[][] matrix)
    {
        long size = (long) matrix.length * matrix[0].length;

        try (INDArray array = Nd4j.create(matrix))
        {
            return array.reshape(size).toFloatVector();
        }
    }

    /**
     * Calculate average (mean) value
     */
    public static float average(float[] vector)
    {
        try (INDArray array = Nd4j.create(vector))
        {
            return array.meanNumber().floatValue();
        }
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