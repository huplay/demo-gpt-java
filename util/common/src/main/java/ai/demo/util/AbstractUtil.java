package ai.demo.util;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public abstract class AbstractUtil implements Utility
{
    @Override
    public float max(float[] vector)
    {
        float max = Float.NEGATIVE_INFINITY;

        for (float value : vector)
        {
            if (value > max)
            {
                max = value;
            }
        }

        return max;
    }

    @Override
    public float max(List<IndexedValue> vector)
    {
        float max = Float.NEGATIVE_INFINITY;

        for (IndexedValue indexedValue : vector)
        {
            if (indexedValue.getValue() > max)
            {
                max = indexedValue.getValue();
            }
        }

        return max;
    }

    @Override
    public float[] normalize(float[] vector, float epsilon)
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

    @Override
    public float averageDiff(float[] values, float average, float epsilon)
    {
        float[] squareDiff = new float[values.length];

        for (int i = 0; i < values.length; i++)
        {
            float diff = values[i] - average;
            squareDiff[i] = diff * diff;
        }

        float averageSquareDiff = average(squareDiff);

        return (float) Math.sqrt(averageSquareDiff + epsilon);
    }


    /**
     * Sort values to reversed order and filter out the lowest values (retain the top [count] values)
     */
    public List<IndexedValue> reverseAndFilter(float[] values, int count)
    {
        TreeSet<IndexedValue> indexedValues = new TreeSet<>(new IndexedValue.ReverseComparator());
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
}
