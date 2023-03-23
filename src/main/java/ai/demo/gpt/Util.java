package ai.demo.gpt;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static java.lang.Math.sqrt;

public class Util
{
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_MAX;

    /**
     * Vector to vector addition
     */
    public static float[] addVectors(float[] vector1, float[] vector2)
    {
        float[] result = new float[vector1.length];

        for (int i = 0; i < vector1.length; i += SPECIES.length())
        {
            VectorMask<Float> mask = SPECIES.indexInRange(i, vector1.length);
            FloatVector first = FloatVector.fromArray(SPECIES, vector1, i, mask);
            FloatVector second = FloatVector.fromArray(SPECIES, vector2, i, mask);
            first.add(second).intoArray(result, i, mask);
        }

        return result;
    }

    /**
     * Dot product calculation (multiplying vector by vector)
     */
    public static float dotProduct(float[] a, float[] b)
    {
        var upperBound = SPECIES.loopBound(a.length);
        var sum = FloatVector.zero(SPECIES);
        var i = 0;
        for (; i < upperBound; i += SPECIES.length()) {
            // FloatVector va, vb, vc
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            sum = va.fma(vb, sum);
        }
        var c = sum.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) { // Cleanup loop
            c += a[i] * b[i];
        }
        return c;
    }

    /**
     * Multiply vector by a scalar
     */
    public static float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        float[] result = new float[vector.length];
        System.out.println(SPECIES.length());
        for (int i = 0; i < vector.length; i += SPECIES.length())
        {
            VectorMask<Float> mask = SPECIES.indexInRange(i, vector.length);
            FloatVector floatVector = FloatVector.fromArray(SPECIES, vector, i, mask);
            floatVector.mul(scalar).intoArray(result, i, mask);
        }

        return result;
    }

    /**
     * Multiply vector by matrix
     */
    public static float[] multiplyVectorByMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix[0].length];

        for (int col = 0; col < matrix[0].length; col++)
        {
            ret[col] = dotProduct(vector, getColumn(matrix, col));
        }

        return ret;
    }

    // TODO: If the matrix would be stored as a transposed matrix (first index is the column),
    // then this method would be obsolete. I guess this is the reason why I had to transpose some weight files from pythorch
    public static float[] getColumn(float[][] matrix, int index)
    {
        float[] column = new float[matrix.length];

        for (int i = 0; i < matrix.length; i++)
        {
            column[i] = matrix[i][index];
        }

        return column;
    }

    /**
     * Multiply vector by transposed matrix
     */
    public static float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix[0].length];

        for (int col = 0; col < matrix[0].length; col++)
        {
            ret[col] = dotProduct(vector, matrix[col]);
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

    private static float average(float[] vector)
    {
        double sum = 0;

        for (float value : vector)
        {
            sum = sum + value;
        }

        return (float) sum / vector.length;
    }

    private static float averageDiff(float[] values, float average, float epsilon)
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
}
