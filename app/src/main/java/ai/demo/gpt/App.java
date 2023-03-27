package ai.demo.gpt;

import ai.demo.util.Util;

import java.io.*;
import java.util.Collections;
import java.util.List;

public class App
{
    public static PrintStream OUT;

    public static void main(String... args) throws Exception
    {
        OUT = new PrintStream(System.out, true, "utf-8");

        OUT.println("  _____________________________      ___");
        OUT.println(" /  _____/\\______   \\__    ___/   __|  /____   _____   ____");
        OUT.println("/   \\  ___ |     ___/ |    |     / __ |/ __ \\ /     \\ /  _ \\");
        OUT.println("\\    \\_\\  \\|    |     |    |    / /_/ \\  ___/|  Y Y  (  <_> )");
        OUT.println(" \\________/|____|     |____|    \\_____|\\_____>__|_|__/\\____/");
        OUT.println("Util: " + Util.getUtilName() + "\n");

        try
        {
            Arguments arguments = readArguments(args);

            OUT.println("Model: " + arguments.getPath() + "/" + arguments.getName());

            Settings settings = new Settings(arguments);

            OUT.println("Number of parameters: " + Math.round(settings.getParameterSize() / 1000000d) + " M");
            OUT.println("Maximum length of generated text: " + arguments.getLengthLimit());
            OUT.println("Output is selected from the best " + arguments.getTopK() + " tokens (topK)");

            OUT.print("\nLoading trained parameters... ");
            Tokenizer tokenizer = new Tokenizer("GPT-2");
            ParameterReader parameterReader = new ParameterReader(arguments.getModelPath(), settings);
            Transformer transformer = new Transformer(settings, tokenizer, parameterReader);
            OUT.print("Done.");

            while (true)
            {
                // Read the input text
                OUT.print("\n\nInput text: ");
                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                String input = reader.readLine();

                // If the input starts with "//", the same session will be continued, otherwise delete the stored state
                if (input != null && input.startsWith("//")) input = input.substring(2);
                else transformer.clear();

                // Split the input text into tokens
                List<Integer> inputTokens = tokenizer.encode(input);

                // Use the Transformer
                List<Integer> outputTokens = transformer.executeAll(inputTokens);

                // Convert the output to text and print it
                String response = tokenizer.decode(outputTokens);
                print(response, outputTokens, tokenizer);
            }
        }
        catch (Exception e)
        {
            OUT.println("\nERROR: " + e.getMessage());
        }
    }

    private static void print(String response, List<Integer> outputTokens, Tokenizer tokenizer)
    {
        // The response was printed token by token, but for multi-token characters only "�" will be displayed

        // Here we recreate the token by token decoded response (which wasn't returned)
        StringBuilder tokenByTokenResponse = new StringBuilder();
        for (int token: outputTokens)
        {
            tokenByTokenResponse.append(tokenizer.decode(Collections.singletonList(token)));
        }

        // If the token by token decoded result is different to the final decoded result, print the corrected version
        if ( ! tokenByTokenResponse.toString().equals(response))
        {
            OUT.print("\nCorrected unicode response:\n" + response);
        }
    }

    private static Arguments readArguments(String[] args) throws Exception
    {
        // Default values
        if (args == null || args.length == 0)
        {
            throw new Exception("The first parameter should be the model name.");
        }

        String name = args[0];

        String path = "models";
        int maxLength = 25;
        int topK = 40;

        if (args.length > 1)
        {
            // Iterate over the passed parameters and override the default values
            for (int i = 1; i < args.length; i++)
            {
                String[] parts = args[i].split("=");
                if (parts.length == 2)
                {
                    String param = parts[0].toLowerCase();
                    String value = parts[1];

                    if (param.equals("path")) path = value;
                    if (param.equals("maxlength")) maxLength = readInt(value, maxLength);
                    else if (param.equals("topk")) topK = readInt(value, topK);
                }
                else OUT.println("\nWARNING: Unrecognisable argument: " + args[i] + "\n");
            }
        }

        return new Arguments(name, path, maxLength, topK);
    }

    private static int readInt(String value, int defaultValue)
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            OUT.println("\nWARNING: The provided value can't be converted to integer (" + value
                    + "). Default value will be used.\n");
        }
        return defaultValue;
    }

    public static class Arguments
    {
        private final String name;
        private final String path;
        private final int lengthLimit;
        private final int topK;

        public Arguments(String name, String path, int lengthLimit, int topK)
        {
            this.name = name;
            this.path = path;
            this.lengthLimit = lengthLimit;
            this.topK = topK;
        }

        public String getName()
        {
            return name;
        }

        public String getPath()
        {
            return path;
        }

        public int getLengthLimit()
        {
            return lengthLimit;
        }

        public int getTopK()
        {
            return topK;
        }

        public String getModelPath()
        {
            return path + "/" + name;
        }
    }
}
