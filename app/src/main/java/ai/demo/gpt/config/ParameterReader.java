package ai.demo.gpt.config;

import ai.demo.util.Util;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * Reader of the trained parameters
 */
public class ParameterReader
{
    private final String modelPath;
    private final Settings settings;

    public ParameterReader(String modelPath, Settings settings)
    {
        this.modelPath = modelPath + "/parameters";
        this.settings = settings;
    }

    public float[] readVector(String name, int size)
    {
        return read(name, size);
    }

    public float[][] readMatrix(String name, int rows, int cols)
    {
        return readMatrix(name, rows, cols, false);
    }

    public float[][] readWeights(String name, int rows, int cols)
    {
        return readMatrix(name, rows, cols, true);
    }

    private float[][] readMatrix(String name, int rows, int cols, boolean isWeight)
    {
        float[] numbers = read(name, rows * cols);

        return toMatrix(numbers, rows, cols, isWeight && settings.isWeightsTransposed());
    }

    private float[] read(String name, int size)
    {
        List<File> files = findFiles(name, size);
        if (files == null) return null;

        float[] array = new float[size];

        int offset = 0;

        for (File file : files)
        {
            int length = (int) file.length() / 4;

            try (FileInputStream stream = new FileInputStream(file))
            {
                FileChannel inChannel = stream.getChannel();

                ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

                buffer.order(settings.getByteOrder());
                FloatBuffer floatBuffer = buffer.asFloatBuffer();
                floatBuffer.get(array, offset, length);

                // TODO: FLOAT16 isn't supported here

                offset += length;
            }
            catch (Exception e)
            {
                throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
            }
        }

        return array;
    }

    private float[][] toMatrix(float[] numbers, int rows, int cols, boolean isTranspose)
    {
        if (isTranspose)
        {
            float[][] transposed = new float[rows][cols];

            int row = 0;
            int col = 0;
            for (int i = 0; i < numbers.length; i++)
            {
                transposed[row][col] = numbers[i];

                row++;

                if (row == rows)
                {
                    row = 0;
                    col++;
                }
            }

            return transposed;
        }
        else
        {
            return Util.splitVector(numbers, rows);
        }
    }

    private List<File> findFiles(String name, int size)
    {
        String mappedName = settings.getFileMappings().get(name);

        if (mappedName == null) mappedName = name + ".dat";
        else if (mappedName.equalsIgnoreCase("<null>")) return null;

        List<File> files = new ArrayList<>();
        long sumSize = 0;

        String fileName = modelPath + "/" + mappedName;
        File file = new File(fileName);

        if (file.exists())
        {
            files.add(file);
            sumSize = file.length();
        }
        else
        {
            // Handling files split into parts
            int i = 1;
            while (true)
            {
                File partFile = new File(fileName + ".part" + i);

                if (partFile.exists())
                {
                    files.add(partFile);
                    sumSize += partFile.length();
                }
                else break;

                i++;
            }

            if (files.isEmpty())
            {
                throw new RuntimeException("Parameter file not found: " + fileName);
            }
        }

        checkSize(files, size, sumSize);

        return files;
    }

    private void checkSize(List<File> files, long expectedSize, long actualSize)
    {
        int numberSize = settings.getDataType().equals("FLOAT16") ? 2 : 4;

        if (actualSize != expectedSize * numberSize)
        {
            throw new RuntimeException("The size of the file(s) (" + files + ", " + actualSize +
                    ") is incorrect. Expected: " + expectedSize * numberSize);
        }
    }
}
