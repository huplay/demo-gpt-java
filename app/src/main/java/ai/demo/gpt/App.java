package ai.demo.gpt;

import ai.demo.gpt.config.Arguments;
import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;
import ai.demo.gpt.position.PositionEmbedder;
import ai.demo.gpt.tokenizer.Tokenizer;
import ai.demo.util.Util;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class App
{
    public static final PrintStream OUT = getPrintStream();
    public static final Util UTIL = new Util();

    public static void main(String... args) throws Exception
    {
        logo();

        Arguments arguments = readArguments(args);
        Settings settings = new Settings(arguments);

        OUT.print("\nLoading parameters... ");

        Tokenizer tokenizer = Tokenizer.getInstance(settings);
        ParameterReader parameterReader = new ParameterReader(settings);
        PositionEmbedder positionEmbedder = PositionEmbedder.getInstance(settings, parameterReader);
        Transformer transformer = new Transformer(settings, tokenizer, parameterReader, positionEmbedder);

        OUT.print("Done.");

        int pos = 0;
        int lastToken = settings.getEndOfTextToken();

        while (true)
        {
            // Read the input text
            String inputText = input();

            List<Integer> inputTokens = new ArrayList<>();

            // If the input starts with "+" continue the same session
            if (inputText.equals("+")) inputTokens.add(lastToken);
            else if (inputText.startsWith("+"))
            {
                inputTokens.addAll(tokenizer.encode(inputText.substring(1)));
            }
            else
            {
                // Convert the input text into list of tokens
                inputTokens = tokenizer.encode(inputText);

                // Clear the transformer's stored values
                pos = 0;
                transformer.clear();
            }

            // Use the Transformer
            List<Integer> outputTokens = transformer.process(inputTokens, pos);

            // Convert the output to text and print it
            String response = tokenizer.decode(outputTokens);
            print(response, outputTokens, tokenizer);

            pos += outputTokens.size();
            lastToken = outputTokens.get(outputTokens.size() - 1);
        }
    }

    private static void logo()
    {
        OUT.println("  _____________________________      ___");
        OUT.println(" /  _____/\\______   \\__    ___/   __|  /____   _____   ____");
        OUT.println("/   \\  ___ |     ___/ |    |     / __ |/ __ \\ /     \\ /  _ \\");
        OUT.println("\\    \\_\\  \\|    |     |    |    / /_/ \\  ___/|  Y Y  (  <_> )");
        OUT.println(" \\________/|____|     |____|    \\_____|\\_____>__|_|__/\\____/");
        OUT.println("Util: " + UTIL.getUtilName() + "\n");
    }

    private static String input() throws IOException
    {
        OUT.print("\n\nInput text: ");
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        return reader.readLine();
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
        boolean isCalculationOnly = false;

        if (args.length > 1)
        {
            // Iterate over the passed parameters and override the default values
            for (int i = 1; i < args.length; i++)
            {
                String[] parts = args[i].split("=");
                if (parts.length == 2)
                {
                    String param = parts[0];
                    String value = parts[1];

                    if (param.equals("path")) path = value;
                    else if (param.equals("max")) maxLength = readInt(value, maxLength);
                    else if (param.equals("topk")) topK = readInt(value, topK);
                    else if (param.equals("calc")) isCalculationOnly = true;
                }
                else OUT.println("\nWARNING: Unrecognisable argument: " + args[i] + "\n");
            }
        }

        return new Arguments(name, path, maxLength, topK, isCalculationOnly);
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

    private static PrintStream getPrintStream()
    {
        try
        {
            return new PrintStream(System.out, true, "utf-8");
        }
        catch (Exception e)
        {
            return System.out;
        }
    }
}
