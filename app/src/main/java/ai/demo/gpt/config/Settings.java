package ai.demo.gpt.config;

import java.io.*;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import static ai.demo.gpt.App.OUT;
import static java.nio.ByteOrder.BIG_ENDIAN;
import static java.nio.ByteOrder.LITTLE_ENDIAN;

/**
 * Holder of the configuration stored in the model.properties file
 */
public class Settings
{
    public static final String ATTENTION_GLOBAL = "global";
    public static final String ATTENTION_LOCAL = "local";
    public static final String ATTENTION_NONE = "none";

    private final Arguments arguments;

    private final String tokenizer;
    private final String tokenizerConfig;
    private final int tokenCount;
    private final int endOfTextToken;
    private final int maxLength;

    private final String positionEncoder;

    private final boolean isInputNormalization;
    private final boolean isPreNormalization;

    private final int hiddenSize;
    private final int feedForwardSize;
    private final int decoderCount;
    private final int headCount;
    private final float attentionDividend;
    private final String[] attentionType;
    private final float epsilon;
    private final int localAttentionSize;

    public final Map<String, String> fileMappings = new HashMap<>();
    public final Map<String, String> matrixOrders = new HashMap<>();
    private final String dataType;
    private final ByteOrder byteOrder;
    private final boolean isWeightsTransposed;
    private final boolean isMergedQKV;
    private final boolean isMergedQKVSplitHeadFirst;
    private final int memorySaverDecoders;

    public Settings(Arguments arguments) throws Exception
    {
        this.arguments = arguments;

        // Read all properties from the model.properties file
        Map<String, String> properties = readProperties(getModelPath() + "model.properties");

        tokenizer = getProperty(properties, "tokenizer");
        tokenizerConfig = getProperty(properties, "tokenizer.config");
        tokenCount = getIntProperty(properties, "token.count");
        endOfTextToken = getIntProperty(properties, "end.of.text.token");
        maxLength = getIntProperty(properties, "max.length");
        positionEncoder = getProperty(properties, "position.embedding");
        isInputNormalization = "true".equals(getProperty(properties, "input.normalization", true));
        isPreNormalization = "true".equals(getProperty(properties, "pre.normalization", true));
        hiddenSize = getIntProperty(properties, "hidden.size");
        feedForwardSize = getIntProperty(properties, "feedforward.size");
        decoderCount = getIntProperty(properties, "decoder.count");
        headCount = getIntProperty(properties, "attention.head.count");
        attentionDividend = getFloatProperty(properties, "attention.dividend");
        epsilon = getFloatProperty(properties, "epsilon");

        boolean isLocalUsed = false;

        // Collect the attention type for all decoders
        String[] attentionType = new String[decoderCount];
        for (int i = 0; i < decoderCount; i++)
        {
            String type = getProperty(properties, "attention.type." + (i + 1));

            if (!type.equals(ATTENTION_GLOBAL) && !type.equals(ATTENTION_LOCAL) && !type.equals(ATTENTION_NONE))
            {
                throw new Exception("Incorrect attention type: '" + type + "'" + " Possible values: '"
                        + ATTENTION_GLOBAL + "'," + "'" + ATTENTION_LOCAL + "'," + "'" + ATTENTION_NONE + "'.");
            }

            if (type.equals(ATTENTION_LOCAL)) isLocalUsed = true;

            attentionType[i] = type;
        }
        this.attentionType = attentionType;

        if (isLocalUsed) localAttentionSize = toInt(getProperty(properties, "attention.local.size"));
        else localAttentionSize = Integer.MAX_VALUE;

        for (Map.Entry<String, String> entry : properties.entrySet())
        {
            if (entry.getKey().startsWith("file."))
            {
                fileMappings.put(entry.getKey().substring(5), entry.getValue());
            }
            else if (entry.getKey().startsWith("matrix.order."))
            {
                matrixOrders.put(entry.getKey().substring(13), entry.getValue());
            }
        }

        this.dataType = properties.get("data.type");
        this.byteOrder = "LITTLE_ENDIAN".equalsIgnoreCase(properties.get("byte.order")) ? LITTLE_ENDIAN : BIG_ENDIAN;
        this.isWeightsTransposed = "true".equalsIgnoreCase(properties.get("weights.transposed"));
        this.isMergedQKV = "true".equalsIgnoreCase(properties.get("merged.qkv"));
        this.isMergedQKVSplitHeadFirst = "HEAD_FIRST".equalsIgnoreCase(properties.get("merged.qkv.split"));

        String memorySaver = properties.get("memory.saver.decoders");
        this.memorySaverDecoders = memorySaver == null ? 0 : toInt(memorySaver);

        // Print settings
        OUT.println("Model: " + arguments.getPath() + "/" + arguments.getName());
        OUT.print("Number of parameters: " + Math.round(getParameterSize() / 1000000d) + " M");
        OUT.print(" (Hidden size: " + hiddenSize + ", decoders: " + decoderCount);
        OUT.println(", heads: " + headCount + ", head size: " + getHeadSize() +")");

        long cost = getCalculationCost(50, 100);
        OUT.println("Inference calculation cost: " + cost / 1_000_000_000 + " GigaFLOPS");

        if (arguments.isCalculationOnly()) System.exit(0);

        OUT.println("Maximum length of generated text: " + arguments.getLengthLimit());
        OUT.println("Output is selected from the best " + arguments.getTopK() + " tokens (topK)");
    }

    public static Map<String, String> readProperties(String fileName) throws Exception
    {
        Map<String, String> properties = new HashMap<>();

        try (Scanner scanner = new Scanner(new File(fileName)))
        {
            while (scanner.hasNextLine())
            {
                String line = scanner.nextLine();
                if (line != null && !line.trim().equals("") && !line.startsWith("#"))
                {
                    String[] parts = line.split("=");
                    if (parts.length == 2)
                    {
                        properties.put(parts[0].trim(), parts[1].trim());
                    }
                    else
                    {
                        OUT.println("\nWARNING: Unrecognizable properties line: (" + fileName + "): " + line);
                    }
                }
            }
        }
        catch (IOException e)
        {
            throw new Exception("Cannot read model.properties file: " + fileName);
        }

        return properties;
    }

    public long getCalculationCost(int inputLength, int outputLength)
    {
        long normCost = 4 * hiddenSize; // TODO: Sqrt averagediff + epsion is missing, but not much

        long layer1Cost = dotProductCost(hiddenSize) * feedForwardSize // weights
                + feedForwardSize // biases
                + 8 * feedForwardSize; // gelu

        long layer2Cost = dotProductCost(feedForwardSize) * hiddenSize + hiddenSize;

        long decoderCost = 2 * normCost // norm cost
                + getAverageAttentionCost(inputLength + outputLength) // Attention mechanism
                + layer1Cost + layer2Cost // Feed forward block
                + hiddenSize * 2; // Residual connections

        long determineOutputCost = dotProductCost(hiddenSize) * tokenCount + softmaxCost(40);

        // TODO: Order and weighted random pick is missing

        return hiddenSize * (inputLength + outputLength)
                + decoderCost * decoderCount * inputLength + outputLength
                + normCost * outputLength + 1 // Final normalization
                + determineOutputCost * (outputLength + 1);
    }

    private long dotProductCost(long size)
    {
        return 2 * size - 1;
    }

    private long softmaxCost(int size)
    {
        return 4 * size;

        // Exp calculation was added as multiplication
    }

    private long getAverageAttentionCost(int length)
    {
        long QKVCost = dotProductCost(hiddenSize) * 3 * hiddenSize;

        long attentionCost = dotProductCost(hiddenSize)
                + 1 // Attention dividend
                + softmaxCost(hiddenSize)
                + 2 * hiddenSize;

        return QKVCost + attentionCost * length / 2;
    }

    public long getParameterSize()
    {
        long wteSize = (long) tokenCount * hiddenSize;

        long wpeSize = positionEncoder.equals("LEARNED") ? (long) maxLength * hiddenSize : 0;

        long finalNormSize = (long) hiddenSize * 2;

        return wteSize + wpeSize + (getDecoderParameterSize() * decoderCount) + finalNormSize;
    }

    private long getDecoderParameterSize()
    {
        long qkvSize = ((long) hiddenSize * hiddenSize + hiddenSize) * 3;
        long projSize = (long) hiddenSize * hiddenSize + hiddenSize;
        long normSize = (long) hiddenSize * 4;
        long layer1Size = ((long) hiddenSize * feedForwardSize + feedForwardSize);
        long layer2Size = (long) hiddenSize * feedForwardSize + hiddenSize;

        return qkvSize + projSize + normSize + layer1Size + layer2Size;
    }

    private int getIntProperty(Map<String, String> properties, String key) throws Exception
    {
        return toInt(getProperty(properties, key));
    }

    private float getFloatProperty(Map<String, String> properties, String key) throws Exception
    {
        return toFloat(getProperty(properties, key));
    }

    private String getProperty(Map<String, String> properties, String key) throws Exception
    {
        return getProperty(properties, key, false);
    }

    private String getProperty(Map<String, String> properties, String key, boolean isOptional) throws Exception
    {
        String value = properties.get(key);

        if ( ! isOptional && value == null)
        {
            throw new Exception("Missing entry in the model.properties file: '" + key + "'.");
        }

        return value;
    }

    private int toInt(String value) throws Exception
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to integer (" + value + ").");
        }
    }

    private float toFloat(String value) throws Exception
    {
        try
        {
            return Float.parseFloat(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to float (" + value + ").");
        }
    }

    public String getModelPath()
    {
        return arguments.getPath() + "/" + arguments.getName() + "/";
    }

    public String getParametersPath()
    {
        return getModelPath() + "parameters/";
    }

    public String getTokenizer()
    {
        return tokenizer;
    }

    public String getTokenizerConfig()
    {
        return tokenizerConfig;
    }

    public int getLengthLimit()
    {
        return arguments.getLengthLimit();
    }

    public int getTopK()
    {
        return arguments.getTopK();
    }

    public int getTokenCount()
    {
        return tokenCount;
    }

    public int getEndOfTextToken()
    {
        return endOfTextToken;
    }

    public int getMaxLength()
    {
        return maxLength;
    }

    public String getPositionEncoder()
    {
        return positionEncoder;
    }

    public boolean isInputNormalization()
    {
        return isInputNormalization;
    }

    public boolean isPreNormalization()
    {
        return isPreNormalization;
    }

    public boolean isOutputNormalization()
    {
        return isPreNormalization;
    }

    public int getHiddenSize()
    {
        return hiddenSize;
    }

    public int getFeedForwardSize()
    {
        return feedForwardSize;
    }

    public int getDecoderCount()
    {
        return decoderCount;
    }

    public int getHeadCount()
    {
        return headCount;
    }

    public float getAttentionDividend()
    {
        return attentionDividend;
    }

    public float getEpsilon()
    {
        return epsilon;
    }

    public String[] getAttentionType()
    {
        return attentionType;
    }

    public int getLocalAttentionSize()
    {
        return localAttentionSize;
    }

    public Map<String, String> getFileMappings()
    {
        return fileMappings;
    }

    public String getDataType()
    {
        return dataType;
    }

    public ByteOrder getByteOrder()
    {
        return byteOrder;
    }

    public boolean isWeightsTransposed()
    {
        return isWeightsTransposed;
    }

    public boolean isMergedQKV()
    {
        return isMergedQKV;
    }

    public boolean isMergedQKVSplitHeadFirst()
    {
        return isMergedQKVSplitHeadFirst;
    }

    public int getMemorySaverDecoders()
    {
        return memorySaverDecoders;
    }

    public int getHeadSize()
    {
        return hiddenSize / headCount;
    }

    public boolean hasAttention(int decoderId)
    {
        return ! attentionType[decoderId].equalsIgnoreCase(ATTENTION_NONE);
    }

    public int getMaxAttentionSize(int decoderId)
    {
        return attentionType[decoderId].equalsIgnoreCase(ATTENTION_LOCAL) ? localAttentionSize : Integer.MAX_VALUE;
    }
}
