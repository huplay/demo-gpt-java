package ai.demo.gpt.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class Pair
{
    private final String left;
    private final String right;

    public Pair(String left, String right)
    {
        this.left = left;
        this.right = right;
    }

    /**
     * Split a text into merge pairs
     */
    public static List<Pair> getPairs(List<String> word)
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
     * Find a pair in the merges
     */
    public static Pair findFirstPair(List<Pair> pairs, Map<Pair, Integer> merges)
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

    public String getLeft()
    {
        return left;
    }

    public String getRight()
    {
        return right;
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
