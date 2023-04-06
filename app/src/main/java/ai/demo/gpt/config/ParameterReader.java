package ai.demo.gpt.config;

import ai.demo.util.Util;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * Reader of the trained parameters
 */
public class ParameterReader
{
    private final Settings settings;

    public ParameterReader(Settings settings)
    {
        this.settings = settings;
    }

    public float[] readVector(String name, int size)
    {
        return readVector(name, size, 1, 0);
    }

    public float[] readVector(String name, int size, int segments, int index)
    {
        return read(name, size, segments, index);
    }

    public float[][] readMatrix(String name, int rows, int cols)
    {
        float[] vector = read(name, rows * cols, 1, 0);
        return vector == null ? null : Util.splitVector(vector, rows);
    }

    public float[][] readWeights(String name, int rows, int cols, int segments, int index)
    {
        float[] vector = read(name, rows * cols, segments, index);

        if (vector == null) return null;

        if (settings.isWeightsTransposed())
        {
            return splitVectorTransposed(vector, rows, cols);
        }
        else
        {
            return Util.splitVector(vector, rows);
        }
    }

    private float[] read(String name, int size, int segments, int index)
    {
        int bytes = 4; // TODO: FLOAT16 isn't supported here

        File file = findFile(name, size * segments * bytes);
        if (file == null) return null;

        float[] array = new float[size];

        int position = segments > 1 ? size * index * bytes : 0;

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, position, size * bytes);
            buffer.order(settings.getByteOrder());
            FloatBuffer floatBuffer = buffer.asFloatBuffer();

            floatBuffer.get(array, 0, size);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
        }

        return array;
    }

    private float[][] splitVectorTransposed(float[] numbers, int rows, int cols)
    {
        float[][] matrix = new float[rows][cols];

        int row = 0;
        int col = 0;
        for (int i = 0; i < numbers.length; i++)
        {
            matrix[row][col] = numbers[i];

            row++;

            if (row == rows)
            {
                row = 0;
                col++;
            }
        }

        return matrix;
    }

    private File findFile(String name, int size)
    {
        String mappedName = settings.getFileMappings().get(name);

        if (mappedName == null) mappedName = name + ".dat";
        else if (mappedName.equalsIgnoreCase("<null>")) return null;

        String fileName = settings.getParametersPath() + mappedName;
        File file = new File(fileName);

        if ( ! file.exists())
        {
            // Handling files split into parts
            List<File> partFiles = new ArrayList<>();

            int i = 1;
            while (true)
            {
                File partFile = new File(fileName + ".part" + i);

                if (partFile.exists()) partFiles.add(partFile);
                else break;

                i++;
            }

            if (partFiles.isEmpty())
            {
                throw new RuntimeException("Parameter file not found: " + fileName);
            }
            else
            {
                file = mergeAndSaveParts(settings, partFiles, fileName);
            }
        }

        if (file.length() != size)
        {
            throw new RuntimeException("Incorrect file size (" + file.length() + "). Expected: " + size);
        }

        return file;
    }

    private File mergeAndSaveParts(Settings settings, List<File> partFiles, String fileName)
    {
        File file = new File(fileName);

        try
        {
            DataOutputStream output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));

            for (File partFile : partFiles)
            {
                float[] array = new float[(int) partFile.length() / 4]; // TODO: Only FLOAT32 is supported here

                try (FileInputStream stream = new FileInputStream(partFile))
                {
                    FileChannel inChannel = stream.getChannel();
                    ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());
                    buffer.order(settings.getByteOrder());
                    FloatBuffer floatBuffer = buffer.asFloatBuffer();

                    floatBuffer.get(array);
                }
                catch (Exception e)
                {
                    throw new RuntimeException("Parameter file read error. (" + partFile.getName() + ")");
                }

                for (float floatValue : array)
                {
                    output.writeFloat(floatValue);
                }
            }

            output.close();
        }
        catch (IOException e)
        {
            throw new RuntimeException("Can't create concatenated file (" + fileName + ") Exception: " + e.getMessage());
        }

        return file;
    }
}
