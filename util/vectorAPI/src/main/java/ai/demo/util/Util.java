package ai.demo.util;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Util extends AbstractUtil
{
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_MAX;

    @Override
    public String getUtilName()
    {
        return "Java Vector API (" + SPECIES.vectorBitSize() + " bit)";
    }

    @Override
    public float[] addVectors(float[] vector1, float[] vector2)
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

    @Override
    public float dotProduct(float[] a, float[] b)
    {
        var upperBound = SPECIES.loopBound(a.length);
        var sum = FloatVector.zero(SPECIES);

        var i = 0;
        for (; i < upperBound; i += SPECIES.length())
        {
            // FloatVector va, vb, vc
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            sum = va.fma(vb, sum);
        }

        var c = sum.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++)
        { // Cleanup loop
            c += a[i] * b[i];
        }
        return c;
    }

    @Override
    public float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        float[] result = new float[vector.length];

        for (int i = 0; i < vector.length; i += SPECIES.length())
        {
            VectorMask<Float> mask = SPECIES.indexInRange(i, vector.length);
            FloatVector floatVector = FloatVector.fromArray(SPECIES, vector, i, mask);
            floatVector.mul(scalar).intoArray(result, i, mask);
        }

        return result;
    }

    @Override
    public float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix.length];

        for (int col = 0; col < matrix.length; col++)
        {
            ret[col] = dotProduct(vector, matrix[col]);
        }

        return ret;
    }

    @Override
    public float[][] splitVector(float[] vector, int count)
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

    @Override
    public float[] flattenMatrix(float[][] matrix)
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

    @Override
    public float average(float[] vector)
    {
        double sum = 0;

        for (float value : vector)
        {
            sum = sum + value;
        }

        return (float) sum / vector.length;
    }
}