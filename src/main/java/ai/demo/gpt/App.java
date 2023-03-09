package ai.demo.gpt;

import java.io.*;
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
        OUT.println(" \\________/|____|     |____|    \\_____|\\_____>__|_|__/\\____/\n");

        try
        {
            Arguments arguments = readArguments(args);

            OUT.println("Path: " + arguments.getPath());

            Settings settings = new Settings(arguments.getPath(), arguments.getMaxLength(), arguments.getTopK());

            OUT.println("Number of parameters: " + Math.round(settings.getParameterSize() / 1000000d) + " M");
            OUT.println("Maximum length of generated text: " + arguments.getMaxLength());
            OUT.println("Output is selected from the best " + arguments.getTopK() + " tokens (topK)");

            OUT.print("\nLoading trained parameters... ");
            Tokenizer tokenizer = new Tokenizer(arguments.getPath());
            Transformer transformer = new Transformer(settings, tokenizer);
            OUT.println("Done.");

            while (true)
            {
                // Read input text
                OUT.print("\nInput text: ");
                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                String input = reader.readLine();

                // If the input starts with "//", the same session will be continued, otherwise the state is cleared
                if (input != null && input.startsWith("//")) input = input.substring(2);
                else transformer.clear();

                // Split the input text into tokens
                List<Integer> inputTokens = tokenizer.encode(input);

                // Use the Transformer
                List<Integer> outputTokens = transformer.processTokens(inputTokens);

                // Convert the output to text and print it
                String response = tokenizer.decode(outputTokens);
                OUT.println(/*response*/); // Commented out because we printed already the text (token by token)
            }
        }
        catch (Exception e)
        {
            OUT.println("\nERROR: " + e.getMessage());
        }
    }

    private static Arguments readArguments(String[] args) throws Exception
    {
        // Default values
        if (args == null || args.length == 0)
        {
            throw new Exception("The first parameter should be the path of the model parameters.");
        }

        String path = args[0];

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

                    if (param.equals("maxlength")) maxLength = readInt(value, maxLength);
                    else if (param.equals("topk")) topK = readInt(value, topK);
                }
                else OUT.println("\nWARNING: Unrecognisable argument: " + args[i] + "\n");
            }
        }

        return new Arguments(path, maxLength, topK);
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

    private static class Arguments
    {
        private final String path;
        private final int maxLength;
        private final int topK;

        public Arguments(String path, int maxLength, int topK)
        {
            this.path = path;
            this.maxLength = maxLength;
            this.topK = topK;
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
    }
}
