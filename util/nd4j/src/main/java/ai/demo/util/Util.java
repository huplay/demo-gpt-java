package ai.demo.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Util extends AbstractUtil
{
    @Override
    public String getUtilName()
    {
        return "ND4j";
    }

    @Override
    public float[] addVectors(float[] vector1, float[] vector2)
    {
        try (INDArray array1 = Nd4j.create(vector1);
             INDArray array2 = Nd4j.create(vector2))
        {
            return array1.add(array2).toFloatVector();
        }
    }

    @Override
    public float dotProduct(float[] vector1, float[] vector2)
    {
        try (INDArray array1 = Nd4j.create(vector1);
             INDArray array2 = Nd4j.create(vector2))
        {
            return array1.mmul(array2).getFloat(0);
        }
    }

    @Override
    public float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        try (INDArray array = Nd4j.create(vector))
        {
            return array.mul(scalar).toFloatVector();
        }
    }

    @Override
    public float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[][] array = new float[1][vector.length];
        array[0] = vector;

        try (INDArray array1 = Nd4j.create(array);
             INDArray array2 = Nd4j.create(matrix))
        {
            return array1.mmul(array2.transpose()).toFloatVector();
        }
    }

    @Override
    public float[][] splitVector(float[] vector, int count)
    {
        try (INDArray array = Nd4j.create(vector))
        {
            return array.reshape(count, vector.length / count).toFloatMatrix();
        }
    }

    @Override
    public float[] flattenMatrix(float[][] matrix)
    {
        long size = (long) matrix.length * matrix[0].length;

        try (INDArray array = Nd4j.create(matrix))
        {
            return array.reshape(size).toFloatVector();
        }
    }

    @Override
    public float average(float[] vector)
    {
        try (INDArray array = Nd4j.create(vector))
        {
            return array.meanNumber().floatValue();
        }
    }
}