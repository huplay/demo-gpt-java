package ai.demo.gpt;

import java.util.*;
import static ai.demo.gpt.App.OUT;
import static ai.demo.gpt.ParameterReader.*;
import static ai.demo.gpt.TransformerUtil.*;

/**
 * Decoder-only Transformer implementation
 */
public class Transformer
{
    private final Settings settings;
    private final Tokenizer tokenizer;
    private final float[][] tokenEmbeddings;
    private final float[][] positionEmbeddings;
    private final float[] normFinalWeights;
    private final float[] normFinalBiases;
    private final TransformerDecoder[] decoders;

    /**
     * Initialization
     */
    public Transformer(Settings settings, Tokenizer tokenizer)
    {
        String path = settings.getPath();
        int hiddenSize = settings.getHiddenSize();

        this.settings = settings;
        this.tokenizer = tokenizer;
        this.tokenEmbeddings = readMatrixFile(path, "input/wte", settings.getTokenCount(), hiddenSize);
        this.positionEmbeddings = readMatrixFile(path, "input/wpe", settings.getContextSize(), hiddenSize);
        this.normFinalWeights = readVectorFile(path, "output/norm.w", hiddenSize);
        this.normFinalBiases = readVectorFile(path, "output/norm.b", hiddenSize);

        // Create the decoder stack
        this.decoders = new TransformerDecoder[settings.getDecoderCount()];
        for (int i = 0; i < settings.getDecoderCount(); i++)
        {
            this.decoders[i] = new TransformerDecoder(i, settings, settings.getAttentionType()[i]);
        }
    }

    /**
     * Transformer token processing logic
     * This method implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     */
    public List<Integer> processTokens(List<Integer> inputTokens)
    {
        int intputSize = inputTokens.size();

        if (intputSize == 0)
        {
            // If the input is empty, use the END_OF_TEXT token as input
            inputTokens.add(settings.getEndOfTextToken());
            intputSize = 1;
        }
        else
        {
            // Iterating over on the input tokens (excluding the last one) and processing these by the transformer
            // We are not interested in the output of the transformer, but the inner state will be stored
            for (int pos = 0; pos < intputSize - 1; pos++)
            {
                OUT.print("."); // Printing a dot to show there is a progress
                processToken(pos, inputTokens.get(pos));
            }
        }

        // Collector of the generated new tokens
        List<Integer> result = new ArrayList<>();

        int nextToken = inputTokens.get(intputSize - 1);

        // Use the transformer again an again to generate new tokens
        for (int pos = intputSize - 1; pos < settings.getMaxLength() + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            float[] output = processToken(pos, nextToken);

            // The output will be the next new token
            nextToken = selectNextToken(output);
            result.add(nextToken);

            // Exit if the END_OF_TEXT token was chosen or the context size is reached
            if (nextToken == settings.getEndOfTextToken() || (intputSize + result.size() >= settings.getContextSize())) break;
        }

        return result;
    }

    private float[] processToken(int pos, int token)
    {
        // Word token embedding
        float[] hiddenState = tokenEmbeddings[token];

        // Position embedding
        hiddenState = Util.addVectors(hiddenState, positionEmbeddings[pos]);

        // Decoder stack
        for (TransformerDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState);
        }

        // Final normalization
        return normalization(hiddenState, normFinalWeights, normFinalBiases, settings.getEpsilon());
    }

    private int selectNextToken(float[] output)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = Util.multiplyVectorByTransposedMatrix(output, tokenEmbeddings);

        // Sort (higher to lower) the result of the dot products, retaining the order (index) of the related token
        List<IndexedValue> orderedLogits = reverseAndFilter(logits, settings.getTopK());

        // Convert the logits to probabilities
        float[] probabilities = softmax(orderedLogits);

        // Pick one token randomly, using a weighted random selection
        int index = weightedRandomPick(probabilities);

        // Lookup the token id
        int selectedTokenId = orderedLogits.get(index).index;

        // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens
        OUT.print(tokenizer.decode(Collections.singletonList(selectedTokenId)));

        return selectedTokenId;
    }

    /**
     * Clear stored values in all decoders to start a new session
     */
    public void clear()
    {
        for (TransformerDecoder decoder : decoders)
        {
            decoder.clear();
        }
    }
}
