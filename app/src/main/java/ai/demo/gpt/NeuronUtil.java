package ai.demo.gpt;

import ai.demo.gpt.config.Settings;
import ai.demo.util.Util;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

import static java.lang.Math.*;

public class NeuronUtil
{
    private final Settings settings;
    private final int decoderId;

    public NeuronUtil(Settings settings, int decoderId)
    {
        this.settings = settings;
        this.decoderId = decoderId;
    }

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: <a href="https://paperswithcode.com/method/gelu" />
     */
    public float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * value * value * value))));
    }

    /**
     * Applying weights and biases - supporting normal and memory saver implementation
     */
    public float[] applyParams(float[] vector, float[][] weights, float[] biases, String name)
    {
        if (weights != null)
        {
            return applyParameters(vector, weights, biases);
        }
        else
        {
            return applyParametersOnDisk(vector, biases, name);
        }
    }

    /**
     * Applying weights and biases
     */
    private float[] applyParameters(float[] vector, float[][] weights, float[] biases)
    {
        float[] result = Util.multiplyVectorByMatrix(vector, weights);

        if (biases != null)
        {
            result = Util.addVectors(result, biases);
        }

        return result;
    }

    /**
     * Applying weights and biases - memory saver implementation: weights are read directly from disk
     */
    private float[] applyParametersOnDisk(float[] vector, float[] biases, String name)
    {
        float[] result = multiplyVectorByMatrixOnDisk(vector, name, settings);

        if (biases != null)
        {
            result = Util.addVectors(result, biases);
        }

        return result;
    }

    private float[] multiplyVectorByMatrixOnDisk(float[] vector, String name, Settings settings)
    {
        int cols = name.equals("mlp.layer1") ? settings.getFeedForwardSize() : settings.getHiddenSize();
        int rows = name.equals("mlp.layer2") ? settings.getFeedForwardSize() : settings.getHiddenSize();

        boolean isQueryKeyValue = name.equals("att.query") || name.equals("att.key") || name.equals("att.value");

        String parameterName = "decoder" + (decoderId + 1) + "/" + name + ".w";
        int index = 0;

        if (isQueryKeyValue && settings.isQueryKeyValueMerged())
        {
            parameterName = "decoder" + (decoderId + 1) + "/att.query.key.value.w";

            if (name.equals("att.key")) index = rows * settings.getHiddenSize();
            if (name.equals("att.value")) index = rows * settings.getHiddenSize() * 2;
        }

        String mappedName = settings.getFileMappings().get(parameterName);
        if (mappedName == null) mappedName = parameterName + ".dat";

        String fileName = settings.getParametersPath() + mappedName;

        File file = new File(fileName);

        float[] result = new float[cols];

        for (int i = 0; i < cols; i++)
        {
            float[] weights = new float[rows];

            try (FileInputStream stream = new FileInputStream(file))
            {
                FileChannel inChannel = stream.getChannel();

                ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

                buffer.order(settings.getByteOrder());
                FloatBuffer floatBuffer = buffer.asFloatBuffer();
                floatBuffer.get(index, weights, 0, rows);

                // TODO: FLOAT16 isn't supported here

                result[i] = Util.dotProduct(vector, weights);

                index += rows;
            }
            catch (Exception e)
            {
                throw new RuntimeException("Parameter file read error. (" + fileName + ")");
            }
        }

        return result;
    }
}
