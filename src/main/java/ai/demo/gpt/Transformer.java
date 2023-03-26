package ai.demo.gpt;

import java.util.*;
import static ai.demo.gpt.App.OUT;
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
    public Transformer(Settings settings, Tokenizer tokenizer, ParameterReader parameterReader)
    {
        this.settings = settings;
        this.tokenizer = tokenizer;

        int hiddenSize = settings.getHiddenSize();

        this.tokenEmbeddings = parameterReader.readMatrix("input/wte", settings.getTokenCount(), hiddenSize);
        this.positionEmbeddings = parameterReader.readMatrix("input/wpe", settings.getMaxLength(), hiddenSize);
        this.normFinalWeights = parameterReader.readVector("output/norm.w", hiddenSize);
        this.normFinalBiases = parameterReader.readVector("output/norm.b", hiddenSize);

        // Create the decoder stack
        this.decoders = new TransformerDecoder[settings.getDecoderCount()];
        for (int i = 0; i < settings.getDecoderCount(); i++)
        {
            this.decoders[i] = new TransformerDecoder(i, settings, settings.getAttentionType()[i], parameterReader);
        }
    }

    /**
     * Transformer token processing logic
     * This method implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     */
    public List<Integer> executeAll(List<Integer> inputTokens)
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
                execute(pos, inputTokens.get(pos));
            }
        }

        // Collector of the generated new tokens
        List<Integer> result = new ArrayList<>();

        int token = inputTokens.get(intputSize - 1);
        OUT.print(".");

        // Use the transformer again an again to generate new tokens
        for (int pos = intputSize - 1; pos < settings.getLengthLimit() + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            float[] output = execute(pos, token);

            // The output will be the next new token
            token = selectNextToken(output);
            result.add(token);

            // Exit if the END_OF_TEXT token was chosen or the maximum length is reached
            if (token == settings.getEndOfTextToken() || (intputSize + result.size() >= settings.getMaxLength())) break;
        }

        return result;
    }

    private float[] execute(int pos, int token)
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
