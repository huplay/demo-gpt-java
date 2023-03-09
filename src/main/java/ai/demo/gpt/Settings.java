package ai.demo.gpt;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import static ai.demo.gpt.App.OUT;

public class Settings
{
    public static final String WTE_DAT = "input/wte";
    public static final String WPE_DAT = "input/wpe";
    public static final String FINAL_NORM_W_DAT = "output/norm.w";
    public static final String FINAL_NORM_B_DAT = "output/norm.b";
    public static final String ATT_QUERY_W_DAT = "att.query.w";
    public static final String ATT_QUERY_B_DAT = "att.query.b";
    public static final String ATT_KEY_W_DAT = "att.key.w";
    public static final String ATT_KEY_B_DAT = "att.key.b";
    public static final String ATT_VALUE_W_DAT = "att.value.w";
    public static final String ATT_VALUE_B_DAT = "att.value.b";
    public static final String ATT_PROJ_W_DAT = "att.proj.w";
    public static final String ATT_PROJ_B_DAT = "att.proj.b";
    public static final String ATT_NORM_W_DAT = "att.norm.w";
    public static final String ATT_NORM_B_DAT = "att.norm.b";
    public static final String MLP_LAYER1_W_DAT = "mlp.layer1.w";
    public static final String MLP_LAYER1_B_DAT = "mlp.layer1.b";
    public static final String MLP_LAYER2_W_DAT = "mlp.layer2.w";
    public static final String MLP_LAYER2_B_DAT = "mlp.layer2.b";
    public static final String MLP_NORM_W_DAT = "mlp.norm.w";
    public static final String MLP_NORM_B_DAT = "mlp.norm.b";

    public static final String ATTENTION_GLOBAL = "global";
    public static final String ATTENTION_LOCAL = "local";
    public static final String ATTENTION_NONE = "none";

    private final String path;
    private final int maxLength;
    private final int topK;
    private final int tokenCount;
    private final int endOfTextToken;
    private final int contextSize;
    private final int hiddenSize;
    private final int decoderCount;
    private final int headCount;
    private final int scoreDividend;
    private final String[] attentionType;
    private final int localAttentionSize;
    private final boolean hasAttentionBias;
    private final float epsilon;

    public Settings(String path, int maxLength, int topK) throws Exception
    {
        this.path = path;
        this.maxLength = maxLength;
        this.topK = topK;

        // Read all properties from the model.properties file
        Map<String, String> properties = new HashMap<>();

        String fileName = path + "/model.properties";
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
                        properties.put(parts[0].toLowerCase().trim(), parts[1].toLowerCase().trim());
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

        // Find the necessary in the collected properties
        tokenCount = toInt(getProperty(properties, "token.count"));
        endOfTextToken = toInt(getProperty(properties, "end.of.text.token"));
        contextSize = toInt(getProperty(properties, "context.size"));
        hiddenSize = toInt(getProperty(properties, "embedding.size"));
        decoderCount = toInt(getProperty(properties, "decoder.count"));
        headCount = toInt(getProperty(properties, "attention.head.count"));
        scoreDividend = toInt(getProperty(properties, "attention.score.dividend"));
        hasAttentionBias = toBoolean(getProperty(properties, "has.attention.bias", true), true);
        epsilon = toFloat(getProperty(properties, "epsilon"));

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
    }

    private String getProperty(Map<String, String> properties, String key) throws Exception
    {
        return getProperty(properties, key, false);
    }

    private String getProperty(Map<String, String> properties, String key, boolean isOptional) throws Exception
    {
        String value = properties.get(key);

        if (!isOptional && value == null)
        {
            throw new Exception("Missing entry in the model.properties file: '" + key + "'.");
        }

        return value;
    }

    public long getParameterSize()
    {
        long wteSize = (long) tokenCount * hiddenSize;
        long wpeSize = (long) contextSize * hiddenSize;
        long finalNormSize = (long) hiddenSize * 2;

        return wteSize + wpeSize + (getDecoderParameterSize() * decoderCount) + finalNormSize;
    }

    private long getDecoderParameterSize()
    {
        long qkvSize = ((long) hiddenSize * hiddenSize + hiddenSize) * 3;
        long projSize = (long) hiddenSize * hiddenSize + hiddenSize;
        long normSize = (long) hiddenSize * 4;
        long layer1Size = ((long) hiddenSize * hiddenSize + hiddenSize) * 4;
        long layer2Size = (long) hiddenSize * hiddenSize * 4 + hiddenSize;

        return qkvSize + projSize + normSize + layer1Size + layer2Size;
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

    private boolean toBoolean(String value, boolean defaultValue) throws Exception
    {
        if (value == null) return defaultValue;

        try
        {
            return Boolean.parseBoolean(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to boolean (" + value + ").");
        }
    }

    public String getPath()
    {
        return path;
    }

    public int getMaxLength()
    {
        return maxLength;
    }

    public int getTopK()
    {
        return topK;
    }

    public int getTokenCount()
    {
        return tokenCount;
    }

    public int getEndOfTextToken()
    {
        return endOfTextToken;
    }

    public int getContextSize()
    {
        return contextSize;
    }

    public int getHiddenSize()
    {
        return hiddenSize;
    }

    public int getDecoderCount()
    {
        return decoderCount;
    }

    public int getHeadCount()
    {
        return headCount;
    }

    public int getScoreDividend()
    {
        return scoreDividend;
    }

    public String[] getAttentionType()
    {
        return attentionType;
    }

    public int getLocalAttentionSize()
    {
        return localAttentionSize;
    }

    public boolean hasAttentionBias()
    {
        return hasAttentionBias;
    }

    public float getEpsilon()
    {
        return epsilon;
    }
}
