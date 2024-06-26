package ai.demo.gpt;

import ai.demo.gpt.config.ParameterReader;
import ai.demo.gpt.config.Settings;
import ai.demo.gpt.position.PositionEmbedder;
import ai.demo.gpt.tokenizer.Tokenizer;
import ai.demo.util.IndexedValue;

import java.util.*;
import static ai.demo.gpt.App.OUT;
import static ai.demo.gpt.App.UTIL;
import static ai.demo.gpt.TransformerUtil.*;

/**
 * Decoder-only Transformer implementation
 */
public class Transformer
{
    private final Settings settings;
    private final Tokenizer tokenizer;
    private final PositionEmbedder position;
    private final float[][] tokenEmbeddings;
    private float[] inputNormWeights;
    private float[] inputNormBiases;
    private float[] outputNormWeights;
    private float[] outputNormBiases;
    private final TransformerDecoder[] decoders;

    /**
     * Initialization
     */
    public Transformer(Settings settings, Tokenizer tokenizer, ParameterReader reader, PositionEmbedder position)
    {
        this.settings = settings;
        this.tokenizer = tokenizer;
        this.position = position;

        int hiddenSize = settings.getHiddenSize();
        this.tokenEmbeddings = reader.readMatrix("input/wte", settings.getTokenCount(), hiddenSize);

        if (settings.isInputNormalization())
        {
            this.inputNormWeights = reader.readVector("input/norm.w", hiddenSize);
            this.inputNormBiases = reader.readVector("input/norm.b", hiddenSize);
        }

        if (settings.isOutputNormalization())
        {
            this.outputNormWeights = reader.readVector("output/norm.w", hiddenSize);
            this.outputNormBiases = reader.readVector("output/norm.b", hiddenSize);
        }

        // Create the decoder stack
        this.decoders = new TransformerDecoder[settings.getDecoderCount()];
        for (int i = 0; i < settings.getDecoderCount(); i++)
        {
            this.decoders[i] = new TransformerDecoder(i, settings, reader, position);
        }
    }

    /**
     * Transformer token processing logic
     * This method implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     */
    public List<Integer> process(List<Integer> inputTokens, int startPos)
    {
        List<Integer> result = new ArrayList<>();
        int intputSize = inputTokens.size();

        // Process the input tokens (except the last)
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
                processToken(pos + startPos, inputTokens.get(pos), false);
            }
        }

        // Process the last input token and repeat it with the newly generated tokens
        int token = inputTokens.get(intputSize - 1);
        OUT.println(". "); // Printing something to show there is a progress

        // Use the transformer again an again to generate new tokens
        for (int pos = intputSize - 1; pos < settings.getLengthLimit() + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            float[] hiddenState = processToken(pos + startPos, token, true);

            token = determineOutputToken(hiddenState);
            result.add(token);

            // Exit if the END_OF_TEXT token was chosen or the maximum length is reached
            if (token == settings.getEndOfTextToken()) break;
            if (intputSize + result.size() + startPos >= settings.getMaxLength()) break;
        }

        return result;
    }

    private float[] processToken(int pos, int token, boolean withOutput)
    {
        // Word token embedding
        float[] hiddenState = tokenEmbeddings[token];

        // Position embedding
        hiddenState = position.toInput(hiddenState, pos);

        // Optional input normalization
        if (settings.isInputNormalization())
        {
            hiddenState = norm(hiddenState, inputNormWeights, inputNormBiases, settings.getEpsilon());
        }

        // Decoder stack
        for (TransformerDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, withOutput);
        }

        return hiddenState;
    }

    private int determineOutputToken(float[] hiddenState)
    {
        // Final normalization
        if (settings.isOutputNormalization())
        {
            hiddenState = norm(hiddenState, outputNormWeights, outputNormBiases, settings.getEpsilon());
        }

        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = UTIL.multiplyVectorByTransposedMatrix(hiddenState, tokenEmbeddings);

        // Sort (higher to lower) the result of the dot products, retaining the order (index) of the related token
        List<IndexedValue> orderedLogits = UTIL.reverseAndFilter(logits, settings.getTopK());

        // Convert the logits to probabilities
        float[] probabilities = softmax(orderedLogits);

        // Pick one token randomly, using a weighted random selection
        int index = weightedRandomPick(probabilities);

        // Lookup the token id
        int selectedTokenId = orderedLogits.get(index).getIndex();

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
