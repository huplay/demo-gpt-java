package ai.demo.gpt.tokenizer;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class FileReader
{
    public static void readTokensFile(String fileName, Map<String, Integer> tokenEncoding,
                                      Map<Integer, String> tokenDecoding)
    {
        try (Scanner scanner = new Scanner(new File(fileName)))
        {
            while (scanner.hasNext())
            {
                String first = scanner.next();

                if (first.startsWith("{")) first = first.substring(1);
                if (first.startsWith("\"")) first = first.substring(1);
                if (first.endsWith(":")) first = first.substring(0, first.length() - 1);
                if (first.endsWith("\"")) first = first.substring(0, first.length() - 1);

                first = first.replace("\\\"", "\"");
                first = first.replace("\\'", "'");
                first = first.replace("\\\\", "\\");

                while (true)
                {
                    int i = first.indexOf("\\u");
                    if (i == -1) break;

                    String hex = first.substring(i + 2, i + 6);
                    first = first.replace("\\u" + hex, "" + (char)Integer.parseInt(hex, 16));
                }

                String second = scanner.next();

                if (second.endsWith(",")) second = second.substring(0, second.length() - 1);
                if (second.endsWith("}")) second = second.substring(0, second.length() - 1);

                int value = Integer.parseInt(second);

                tokenDecoding.put(value, first);
                tokenEncoding.put(first, value);
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public static Map<Pair, Integer> readMergesFile(String fileName, boolean isOmitFirstLine)
    {
        Map<Pair, Integer> merges = new HashMap<>(50000);

        try
        {
            File file = new File(fileName);
            FileInputStream inputStream = new FileInputStream(file);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));

            if (isOmitFirstLine) reader.readLine();

            int i = 0;
            while (true)
            {
                String line = reader.readLine();

                if (line == null) break;

                String[] pairs = line.split(" ");
                merges.put(new Pair(pairs[0], pairs[1]), i);

                i++;
            }

            reader.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

        return merges;
    }
}
