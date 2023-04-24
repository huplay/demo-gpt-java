package ai.demo.gpt.config;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertArrayEquals;

public class ParameterReaderTest
{
    @Test
    public void testFloat32Vector() throws Exception
    {
        ParameterReader reader = getReader("FLOAT32_BIG");

        float[] vector = reader.readVector("float32", 12);

        float[] expected = new float[] {
                0.0010871647f, 0.036529385f, -0.06729616f, 1.6416052E-4f, -0.06744395f, -0.07135112f,
                0.50393414f, 0.09172336f, -0.04933988f, 0.0032621801f, 0.045722805f, -0.0068674036f};

        assertArrayEquals(expected, vector, 0);
    }

    @Test
    public void testFloat32Matrix() throws Exception
    {
        ParameterReader reader = getReader("FLOAT32_BIG");

        float[][] matrix = reader.readMatrix("float32", 4, 3);

        float[][] expected = new float[][] {
                {0.0010871647f, 0.036529385f, -0.06729616f},
                {1.6416052E-4f, -0.06744395f, -0.07135112f},
                {0.50393414f, 0.09172336f, -0.04933988f},
                {0.0032621801f, 0.045722805f, -0.0068674036f}};

       assertArrayEquals(expected, matrix);
    }

    @Test
    public void testFloat32Weight() throws Exception
    {
        ParameterReader reader = getReader("FLOAT32_BIG");

        float[][] matrix = reader.readWeights("float32", 4, 3);

        float[][] expected = new float[][] {
                {0.0010871647f, 0.036529385f, -0.06729616f},
                {1.6416052E-4f, -0.06744395f, -0.07135112f},
                {0.50393414f, 0.09172336f, -0.04933988f},
                {0.0032621801f, 0.045722805f, -0.0068674036f}};

        assertArrayEquals(expected, matrix);
    }

    @Test
    public void testFloat32WeightTransposed() throws Exception
    {
        ParameterReader reader = getReader("FLOAT32_BIG_TRANSPOSED");

        float[][] matrix = reader.readWeights("float32", 4, 3);

        float[][] expected = new float[][] {
                {0.0010871647f, -0.06744395f, -0.04933988f},
                {0.036529385f, -0.07135112f, 0.0032621801f},
                {-0.06729616f, 0.50393414f, 0.045722805f},
                {1.6416052E-4f, 0.09172336f, -0.0068674036f}};

        assertArrayEquals(expected, matrix);
    }
/*
    @Test
    public void testFloat32VectorSegment() throws Exception
    {
        ParameterReader reader = getReader("FLOAT32_BIG");

        float[] vector = reader.readVector("float32", 4, 3, 0);

        float[] expected = new float[] {0.0010871647f, 0.036529385f, -0.06729616f, 1.6416052E-4f};
        assertArrayEquals(expected, vector, 0);

        vector = reader.readVector("float32", 4, 3, 1);

        expected = new float[] {-0.06744395f, -0.07135112f, 0.50393414f, 0.09172336f};
        assertArrayEquals(expected, vector, 0);

        vector = reader.readVector("float32", 4, 3, 2);

        expected = new float[] {-0.04933988f, 0.0032621801f, 0.045722805f, -0.0068674036f};
        assertArrayEquals(expected, vector, 0);
    }
*/
    /*
    @Test
    public void testFloat32WeightSegment() throws Exception
    {
        ParameterReader reader = getReader("FLOAT32_BIG");

        float[][] weights = reader.readWeights("float32", 3, 2, 2, 0);

        float[][] expected = new float[][] {
                {0.0010871647f, 0.036529385f},
                {-0.06729616f, 1.6416052E-4f},
                {-0.06744395f, -0.07135112f}};

        assertArrayEquals(expected, weights);

        weights = reader.readWeights("float32", 3, 2, 2, 1);

        expected = new float[][] {
                {0.50393414f, 0.09172336f},
                {-0.04933988f, 0.0032621801f},
                {0.045722805f, -0.0068674036f}};

        assertArrayEquals(expected, weights);
    }
*/
    @Test
    public void testFloat32LittleEndianVector() throws Exception
    {
        ParameterReader reader = getReader("FLOAT32_LITTLE");

        float[] vector = reader.readVector("float32", 3);

        float[] expected = new float[] {-0.06711365f, 0.019478641f, 0.06818082f};

        assertArrayEquals(expected, vector, 0);
    }

    @Test
    public void testFloat16LittleEndianVector() throws Exception
    {
        ParameterReader reader = getReader("FLOAT16_LITTLE");

        float[] vector = reader.readVector("float16", 4);

        float[] expected = new float[] {-0.05130005f, 0.016448975f, 0.005153656f, 0.043701172f};

        assertArrayEquals(expected, vector, 0);
    }

    private ParameterReader getReader(String name) throws Exception
    {
        String path = new File("src/test/resources").getAbsolutePath() + "/";

        Arguments arguments = new Arguments(name, path, 10, 10, false);
        Settings settings = new Settings(arguments);
        return new ParameterReader(settings);
    }
}
