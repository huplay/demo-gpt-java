package ai.demo.gpt.config;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.util.*;

import static ai.demo.gpt.App.UTIL;

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
        return read(name, size);
    }

    public float[][] readMatrix(String name, int rows, int cols)
    {
        float[] vector = read(name, rows * cols);
        return vector == null ? null : UTIL.splitVector(vector, rows);
    }

    public float[][] readWeights(String name, int rows, int cols)
    {
        float[] vector = read(name, rows * cols);

        if (vector == null) return null;

        if (settings.isWeightsTransposed())
        {
            return splitVectorTransposed(vector, rows, cols);
        }
        else
        {
            return UTIL.splitVector(vector, rows);
        }
    }

    private float[] read(String name, int size)
    {
        switch (settings.getDataType())
        {
            case "FLOAT32" : return readFloat32(name, size);
            case "FLOAT16" : return readFloat16(name, size);
            case "BFLOAT16" : return readBFloat16(name, size);
            default : throw new RuntimeException("Unsupported data type (" + settings.getDataType() + ")");
        }
    }

    private float[] readFloat32(String name, int size)
    {
        File file = findFile(name, size * 4);
        if (file == null) return null;

        float[] array = new float[size];

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, (long) size * 4);
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

    private float[] readFloat16(String name, int size)
    {
        File file = findFile(name, size * 2);
        if (file == null) return null;

        short[] array = new short[size];

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, (long) size * 2);
            buffer.order(settings.getByteOrder());
            ShortBuffer shortBuffer = buffer.asShortBuffer();

            shortBuffer.get(array, 0, size);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
        }

        float[] ret = new float[size];

        for (int i = 0; i < size; i++)
        {
            ret[i] = toFloat32(array[i]);
        }

        return ret;
    }

    private float[] readBFloat16(String name, int size)
    {
        File file = findFile(name, size * 2);
        if (file == null) return null;

        short[] array = new short[size];
        //byte[] array = new byte[size * 2];

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, (long) size * 2);
            buffer.order(settings.getByteOrder());
            ShortBuffer shortBuffer = buffer.asShortBuffer();

            shortBuffer.get(array, 0, size);
            //buffer.get(array, 0, size * 2);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
        }

        float[] ret = new float[size];

        for (int i = 0; i < size; i++)
        {
            ret[i] = toFloat32(array[i]);
            //ret[i] = toFullPrecision(array[i], array[i+1]);
        }

        return ret;
    }

    /*private float toFullPrecision(byte first, byte second)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocate(4);

        byteBuffer.put((byte)0);
        byteBuffer.put((byte)0);
        byteBuffer.put(first);
        byteBuffer.put(second);

        byteBuffer.position(0);

        return byteBuffer.getFloat();
    }*/

/*
    private float toFloat32(short value)
    {
        int mantisa = value & 0x03ff;
        int exponent = value & 0x7c00;

        if (exponent == 0x7c00)
        {
            exponent = 0x3fc00;
        }
        else if (exponent != 0)
        {
            exponent += 0x1c000;
            if (mantisa == 0 && exponent > 0x1c400)
            {
                return Float.intBitsToFloat((value & 0x8000) << 16 | exponent << 13 | 0x3ff);
            }
        }
        else if (mantisa != 0)
        {
            exponent = 0x1c400;
            do
            {
                mantisa <<= 1;
                exponent -= 0x400;
            }
            while ((mantisa & 0x400) == 0);

            mantisa &= 0x3ff;
        }

        return Float.intBitsToFloat((value & 0x8000) << 16 | (exponent | mantisa) << 13);
    }*/

    private float toFloat32(short value)
    {
        int signFlag = value & 0b1000_0000_0000_0000; // Extract sign (1st bit)
        int exponent = value & 0b0111_1100_0000_0000; // Extract exponent (5 bits after exponent
        int mantissa = value & 0b0000_0011_1111_1111; // Extract mantissa (last 10 bits)

        if (exponent == 0b0111_1100_0000_0000)
        {
            // Infinity or NaN
            if (mantissa == 0)
            {
                if (signFlag == 0) return Float.POSITIVE_INFINITY;
                else return Float.NEGATIVE_INFINITY;
            }
            else return Float.NaN;
        }
        else if (exponent == 0)
        {
            // Zero or subnormal value
            if (mantissa != 0)
            {
                exponent = 0x1c400;
                do
                {
                    mantissa <<= 1;
                    exponent -= 0b0000_0100_0000_0000;
                }
                while ((mantissa & 0b0000_0100_0000_0000) == 0);

                mantissa &= 0b0000_0011_1111_1111;
            }

            return Float.intBitsToFloat(signFlag << 16 | (exponent | mantissa) << 13);
        }
        else
        {
            // Normal value
            exponent += 0x1c000;
            if (mantissa == 0 && exponent > 0x1c400)
            {
                return Float.intBitsToFloat(signFlag << 16 | exponent << 13 | 0b0000_0011_1111_1111);
            }

            return Float.intBitsToFloat(signFlag << 16 | (exponent | mantissa) << 13);
        }
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
                file = mergeAndSaveParts(partFiles, fileName);
            }
        }

        if (file.length() != size)
        {
            throw new RuntimeException("Incorrect file size (" + file.length() + "). Expected: " + size);
        }

        return file;
    }

    private File mergeAndSaveParts(List<File> partFiles, String fileName)
    {
        File file = new File(fileName);

        try
        {
            DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file.toPath())));

            for (File partFile : partFiles)
            {
                byte[] array = new byte[(int) partFile.length()];

                try (FileInputStream stream = new FileInputStream(partFile))
                {
                    FileChannel inChannel = stream.getChannel();
                    ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

                    buffer.get(array);
                }
                catch (Exception e)
                {
                    throw new RuntimeException("Parameter file read error. (" + partFile.getName() + ")");
                }

                for (byte value : array)
                {
                    output.writeByte(value);
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
