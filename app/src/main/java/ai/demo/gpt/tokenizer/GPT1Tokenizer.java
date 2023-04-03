package ai.demo.gpt.tokenizer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tokenizer which is similar to OpenAI's GPT-1 tokenizer (Not fully compatible, but for most cases should work)
 * https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/openai/tokenization_openai.py#L233
 */
public class GPT1Tokenizer implements Tokenizer
{
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

    public List<Integer> encode(String text)
    {
        if (text == null) return Collections.singletonList(0);

        text = text.replace("—", "-");
        text = text.replace("–", "-");
        text = text.replace("―", "-");
        text = text.replace("…", "...");
        text = text.replace("´", "'");
        //text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
        //text = re.sub(r"\s*\n\s*", " \n ", text)
        //text = re.sub(r"[^\S\n]+", " ", text)
        text = text.toLowerCase().trim();

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
            String word = tokenDecoding.get(token);

            if (word != null)
            {
                if (word.endsWith("</w>"))
                {
                    word = word.substring(0, word.length() - 4) + " ";
                }

                textBuilder.append(word);
            }
        }
        String text = textBuilder.toString();

        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++)
        {
            char chr = text.charAt(i);
            bytes[i] = (chr == ' ') ? 32 : charEncoding.get(text.charAt(i));
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
