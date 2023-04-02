package ai.demo.gpt.tokenizer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class GPT1Tokenizer implements Tokenizer
{
    // https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/openai/tokenization_openai.py#L233
    // OpenAIGPTTokenizer

    private final Map<Character, Byte> charEncoding = new HashMap<>(256);
    private final Map<Integer, Character> charDecoding = new HashMap<>(256);

    private final Map<String, Integer> tokenEncoding = new HashMap<>(50257);
    private final Map<Integer, String> tokenDecoding = new HashMap<>(50257);

    private final Map<Pair, Integer> merges;

    private final Pattern pattern =
            Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    public GPT1Tokenizer(String path)
    {
        addCharRange(0, 'Ā', 'Ġ');
        addCharRange(33, '!', '~');
        addCharRange(127, 'ġ', 'ł');
        addCharRange(161, '¡', '¬');
        addCharRange(173, 'Ń', 'Ń');
        addCharRange(174, '®', 'ÿ');

        FileReader.readTokensFile(path + "/encoder_bpe_40000.json", tokenEncoding, tokenDecoding);
        merges = FileReader.readMergesFile(path + "/vocab_40000.bpe", true);
    }

    private void addCharRange(int pos, char firstChar, char lastChar)
    {
        for (int i = firstChar; i <= lastChar; i++)
        {
            charEncoding.put((char) i, (byte)pos);
            charDecoding.put(pos, (char) i);
            pos++;
        }
    }

    /**
     * Convert text to list of tokens
     */
    public List<Integer> encode(String text)
    {
        if (text == null) return Collections.singletonList(0);

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
                match.append(charDecoding.get(value));
            }

            unicodes.add(match.toString());
        }

        for (String word : unicodes)
        {
            for (String token : bpe(word).split(" "))
            {
                Integer value = tokenEncoding.get(token);
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
            /*String word = tokenDecoding.get(token);

            if (word != null)
            {
                if (word.endsWith("</w>"))
                {
                    word = word.substring(0, word.length() - 4) + " ";
                }

                textBuilder.append(word);
            }*/

            textBuilder.append(tokenDecoding.get(token));
        }
        String text = textBuilder.toString();

        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++)
        {
            bytes[i] = charEncoding.get(text.charAt(i));
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

        List<Pair> pairs = Pair.getPairs(word);

        while (true)
        {
            Pair pair = Pair.findFirstPair(pairs, merges);
            if (pair == null) break;

            List<String> newWord = new ArrayList<>();

            int i = 0;
            while (i < word.size())
            {
                int j = findFromIndex(word, pair.getLeft(), i);

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

                if (word.get(i).equals(pair.getLeft()) && i < word.size() - 1 && word.get(i + 1).equals(pair.getRight()))
                {
                    newWord.add(pair.getLeft() + pair.getRight());
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
                pairs = Pair.getPairs(word);
            }
        }

        return String.join(" ", word);
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
}
