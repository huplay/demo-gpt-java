package ai.demo.gpt;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
        float[] numbers = read(name, rows * cols);

        if (numbers == null) return null;

        boolean isRowOrganised = isRowOrganised(name);

        return toMatrix(numbers, rows, cols, isRowOrganised);
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

    private float[][] toMatrix(float[] numbers, int rows, int cols, boolean isRowOrganised)
    {
        if (isRowOrganised)
        {
            return Util.splitVector(numbers, rows);
        }
        else
        {
            float[][] transposed = new float[rows][cols];

            int n = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed[i][j] = numbers[n];
                    n++;
                }
            }

            return transposed;
        }
    }

    private boolean isRowOrganised(String name)
    {
        for (Map.Entry<String, String> entry : settings.getMatrixOrders().entrySet())
        {
            Pattern pattern = Pattern.compile(entry.getKey().replace(".", "\\.").replace("*", ".*"));
            Matcher matcher = pattern.matcher(name);
            if (matcher.matches())
            {
                return entry.getValue().equalsIgnoreCase("ROW");
            }
        }

        return true;
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
