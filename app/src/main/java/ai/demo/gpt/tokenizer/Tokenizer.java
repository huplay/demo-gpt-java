package ai.demo.gpt.tokenizer;

import ai.demo.gpt.config.Settings;

import java.util.List;

public interface Tokenizer
{
    List<Integer> encode(String text);

    String decode(List<Integer> tokens);

    static Tokenizer getInstance(Settings settings)
    {
        String path = "tokenizerConfig/" + settings.getTokenizerConfig();

        switch (settings.getTokenizer())
        {
            case "GPT-1": return new GPT1Tokenizer(path);
            case "GPT-2": return new GPT2Tokenizer(path);
        }

        throw new RuntimeException("Unknown tokenizer: " + settings.getTokenizer());
    }
}
