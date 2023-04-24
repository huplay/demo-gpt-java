package ai.demo.util;

import java.util.List;

public interface Utility
{
    String getUtilName();

    /**
     * Vector to vector addition
     */
    float[] addVectors(float[] vector1, float[] vector2);

    /**
     * Dot product calculation (multiplying vector by vector)
     */
    float dotProduct(float[] vector1, float[] vector2);

    /**
     * Multiply vector by a scalar
     */
    float[] multiplyVectorByScalar(float[] vector, float scalar);

    /**
     * Multiply vector by transposed matrix
     */
    float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix);

    /**
     * Split a vector to a matrix
     */
    float[][] splitVector(float[] vector, int count);

    /**
     * Merge the rows of a matrix to a single vector
     */
    float[] flattenMatrix(float[][] matrix);

    /**
     * Finds the maximum value in the vector
     */
    float max(float[] vector);

    /**
     * Finds the maximum value in a list of IndexedValue
     */
    float max(List<IndexedValue> vector);

    /**
     * Calculate average (mean) value
     */
    float average(float[] vector);

    /**
     * Calculate the average difference
     */
    float averageDiff(float[] values, float average, float epsilon);

    /**
     * Standard normalization - (value - avg) * sqrt( (value - avg)^2 + epsilon)
     */
    float[] normalize(float[] vector, float epsilon);
}
