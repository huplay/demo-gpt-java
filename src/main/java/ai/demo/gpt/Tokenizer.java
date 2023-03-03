package ai.demo.gpt;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converting the input text to list of tokens (encode) and the list of tokens to output text (decode)
 */
public class Tokenizer
{
    public static final String ENCODER_FILENAME = "encoder.json";
    public static final String VOCAB_FILENAME = "vocab.bpe";

    private final Map<Integer, Character> charEncoding = new HashMap<>(256); // byte_encoder
    private final Map<Character, Byte> charDecoding = new HashMap<>(256); // byte_encoder

    private final Map<Integer, String> tokenEncoding = new HashMap<>(50257); // decoder
    private final Map<String, Integer> tokenDecoding = new HashMap<>(50257); // encoder

    private final Map<Pair, Integer> merges = new HashMap<>(50000); // bpe_ranks

    private final Pattern pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    /**
     * Initialization
     */
    public Tokenizer(String path)
    {
        addCharRange(0, 'Ā', 'Ġ');
        addCharRange(33, '!', '~');
        addCharRange(127, 'ġ', 'ł');
        addCharRange(161, '¡', '¬');
        addCharRange(173, 'Ń', 'Ń');
        addCharRange(174, '®', 'ÿ');

        readEncoderFile(path);
        readVocabFile(path);
    }

    private void addCharRange(int pos, char firstChar, char lastChar)
    {
        for (int i = firstChar; i <= lastChar; i++)
        {
            charEncoding.put(pos, (char) i);
            charDecoding.put((char) i, (byte)pos);
            pos++;
        }
    }

    /**
     * Read the encoder.json file
     */
    private void readEncoderFile(String path)
    {
        try
        {
            String fileName = path + "/" + ENCODER_FILENAME;
            File file = new File(fileName);
            Scanner scanner = new Scanner(file);

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

                tokenEncoding.put(value, first);
                tokenDecoding.put(first, value);
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Read the vocab.bpe file
     */
    private void readVocabFile(String path)
    {
        try
        {
            String fileName = path + "/" + VOCAB_FILENAME;
            File file = new File(fileName);
            FileInputStream inputStream = new FileInputStream(file);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));

            reader.readLine(); // The first line is a comment

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
    }

    /**
     * Convert text to list of tokens
     */
    public List<Integer> encode(String text)
    {
        List<Integer> result = new ArrayList<>();

        Matcher matcher = pattern.matcher(text);
        List<String> unicodes = new ArrayList<>();

        while (matcher.find())
        {
            StringBuilder match = new StringBuilder();

            ByteBuffer buffer = StandardCharsets.UTF_8.encode(matcher.group());
            while (buffer.hasRemaining())
            {
                int value = buffer.get();
                if (value < 0) value = value & 0xff;
                match.append(charEncoding.get(value));
            }

            unicodes.add(match.toString());
        }

        for (String word : unicodes)
        {
            for (String token : bpe(word).split(" "))
            {
                Integer value = tokenDecoding.get(token);
                if (value != null)
                {
                    result.add(value);
                }
            }
        }

        return result;
    }

    /**
     * Convert list of tokens to text
     */
    public String decode(List<Integer> tokens)
    {
        StringBuilder textBuilder = new StringBuilder();
        for (int token : tokens)
        {
            textBuilder.append(tokenEncoding.get(token));
        }
        String text = textBuilder.toString();

        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++)
        {
            bytes[i] = charDecoding.get(text.charAt(i));
        }

        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Byte pair encoding
     */
    public String bpe(String token)
    {
        if (token == null || token.length() < 2) return token;

        List<String> word = new ArrayList<>();
        for (char c : token.toCharArray())
        {
            word.add(String.valueOf(c));
        }

        List<Pair> pairs = getPairs(word);

        while (true)
        {
            Pair pair = findFirstPair(pairs);
            if (pair == null) break;

            List<String> newWord = new ArrayList<>();

            int i = 0;
            while (i < word.size())
            {
                int j = findFromIndex(word, pair.left, i);

                if (j != -1)
                {
                    newWord.addAll(word.subList(i, j));
                    i = j;
                }
                else
                {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                }

                if (word.get(i).equals(pair.left) && i < word.size() - 1 && word.get(i + 1).equals(pair.right))
                {
                    newWord.add(pair.left + pair.right);
                    i = i + 2;
                }
                else
                {
                    newWord.add(word.get(i));
                    i++;
                }
            }

            word = newWord;

            if (word.size() == 1)
            {
                break;
            }
            else
            {
                pairs = getPairs(word);
            }
        }

        return String.join(" ", word);
    }

    /**
     * Split a text into merge pairs
     */
    private List<Pair> getPairs(List<String> word)
    {
        List<Pair> pairs = new ArrayList<>();

        String prev = word.get(0);

        for (String character : word.subList(1, word.size()))
        {
            pairs.add(new Pair(prev, character));
            prev = character;
        }

        return pairs;
    }

    /**
     * Find a String in a list starting from a provided position (from index)
     */
    private int findFromIndex(List<String> input, String find, int from)
    {
        for (int i = from; i < input.size(); i++)
        {
            if (input.get(i).equals(find)) return i;
        }

        return -1;
    }

    /**
     * Find a pair in the merges
     */
    public Pair findFirstPair(List<Pair> pairs)
    {
        int min = Integer.MAX_VALUE;
        Pair minPair = null;

        for (Pair pair : pairs)
        {
            Integer value = merges.get(pair);

            if (value != null && value.compareTo(min) < 0)
            {
                min = value;
                minPair = pair;
            }
        }

        return minPair;
    }

    /**
     * Holder of two string values (pair)
     */
    private static class Pair
    {
        public final String left;
        public final String right;

        public Pair(String left, String right)
        {
            this.left = left;
            this.right = right;
        }

        @Override
        public boolean equals(Object o)
        {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return Objects.equals(left, pair.left) && Objects.equals(right, pair.right);
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(left, right);
        }
    }
}
