package ai.demo.gpt.tokenizer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tokenizer which fully compatible with the OpenAI GPT-2 and GPT-3 tokenizer (used for other models as well)
 */
public class GPT2Tokenizer implements Tokenizer
{
    private final Map<Character, Byte> charEncoding = new HashMap<>(256);
    private final Map<Integer, Character> charDecoding = new HashMap<>(256);

    private final Map<String, Integer> tokenEncoding = new HashMap<>(50257);
    private final Map<Integer, String> tokenDecoding = new HashMap<>(50257);

    private final Map<Pair, Integer> merges;

    private final Pattern pattern =
            Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    public GPT2Tokenizer(String path)
    {
        addCharRange(0, 'Ā', 'Ġ');
        addCharRange(33, '!', '~');
        addCharRange(127, 'ġ', 'ł');
        addCharRange(161, '¡', '¬');
        addCharRange(173, 'Ń', 'Ń');
        addCharRange(174, '®', 'ÿ');

        FileReader.readTokensFile(path + "/encoder.json", tokenEncoding, tokenDecoding);
        merges = FileReader.readMergesFile(path + "/vocab.bpe", true);
    }

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
            for (String token : BytePairEncoding.encode(word, merges).split(" "))
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

    public String decode(List<Integer> tokens)
    {
        StringBuilder textBuilder = new StringBuilder();
        for (int token : tokens)
        {
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

    private void addCharRange(int pos, char firstChar, char lastChar)
    {
        for (int i = firstChar; i <= lastChar; i++)
        {
            charEncoding.put((char) i, (byte)pos);
            charDecoding.put(pos, (char) i);
            pos++;
        }
    }
}
