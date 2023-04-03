package ai.demo.gpt.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Byte pair encoding
 */
public class BytePairEncoding
{
    public static String encode(String token, Map<Pair, Integer> merges)
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
            Pair pair = findFirstPair(pairs, merges);
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
                pairs = getPairs(word);
            }
        }

        return String.join(" ", word);
    }

    private static List<Pair> getPairs(List<String> word)
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

    private static Pair findFirstPair(List<Pair> pairs, Map<Pair, Integer> merges)
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

    private static int findFromIndex(List<String> input, String find, int from)
    {
        for (int i = from; i < input.size(); i++)
        {
            if (input.get(i).equals(find)) return i;
        }

        return -1;
    }
}
