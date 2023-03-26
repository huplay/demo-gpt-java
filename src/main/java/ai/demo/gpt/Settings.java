package ai.demo.gpt;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import static ai.demo.gpt.App.OUT;

public class Settings
{
    public static final String ATTENTION_GLOBAL = "global";
    public static final String ATTENTION_LOCAL = "local";
    public static final String ATTENTION_NONE = "none";

    private final App.Arguments arguments;

    private final int tokenCount;
    private final int endOfTextToken;
    private final int maxLength;
    private final int hiddenSize;
    private final int decoderCount;
    private final int headCount;
    private final int attentionDividend;
    private final String[] attentionType;
    private final float epsilon;
    private final int localAttentionSize;

    public Settings(App.Arguments arguments) throws Exception
    {
        this.arguments = arguments;

        // Read all properties from the model.properties file
        String fileName = arguments.getPath() + "/" + arguments.getName() + "/model.properties";
        Map<String, String> properties = readProperties(fileName);

        // Find the necessary in the collected properties
        tokenCount = getIntProperty(properties, "token.count");
        endOfTextToken = getIntProperty(properties, "end.of.text.token");
        maxLength = getIntProperty(properties, "max.length");
        hiddenSize = getIntProperty(properties, "hidden.size");
        decoderCount = getIntProperty(properties, "decoder.count");
        headCount = getIntProperty(properties, "attention.head.count");
        attentionDividend = getIntProperty(properties, "attention.dividend");
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

        return properties;
    }

    public long getParameterSize()
    {
        long wteSize = (long) tokenCount * hiddenSize;
        long wpeSize = (long) maxLength * hiddenSize;
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

        if (!isOptional && value == null)
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

    public String getPath()
    {
        return arguments.getPath();
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

    public int getAttentionDividend()
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
}
