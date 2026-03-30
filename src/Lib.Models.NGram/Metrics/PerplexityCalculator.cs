using Lib.MathCore;

public class PerplexityCalculator
{
    public float ComputePerplexityBigram(NGramModel model, ReadOnlySpan<int> tokens)
    {
        if (tokens.Length < 2)
        {
            return float.PositiveInfinity;
        }

        double lossSum = 0;
        int count = 0;

        for (int i = 1; i < tokens.Length; i++)
        {
            int[] context = { tokens[i - 1] };
            float[] probs = model.NextTokenScores(context);

            float[] logits = new float[probs.Length];
            for (int j = 0; j < probs.Length; j++)
            {
                logits[j] = MathF.Log(probs[j] + 1e-9f);
            }

            float loss = LossCalculator.CrossEntropyLoss(logits, tokens[i]);

            lossSum += loss;
            count++;
        }

        double averageLoss = lossSum / count;
        return (float)Math.Exp(averageLoss);
    }


    public float ComputePerplexityTrigram(TrigramModel model, ReadOnlySpan<int> tokens)
    {
        if (tokens.Length < 2)
        {
            return float.PositiveInfinity;
        }

        double lossSum = 0;
        int count = 0;

        for (int i = 1; i < tokens.Length; i++)
        {
            float[] probs;

            if (i >= 2)
            {
                int[] context = { tokens[i - 2], tokens[i - 1] };
                probs = model.NextTokenScores(context);
            }
            else
            {
                int[] context = { tokens[i - 1] };
                probs = model.bigramModel.NextTokenScores(context);
            }

            float[] logits = new float[probs.Length];
            for (int j = 0; j < probs.Length; j++)
            {
                logits[j] = MathF.Log(probs[j] + 1e-9f);
            }

            float loss = LossCalculator.CrossEntropyLoss(logits, tokens[i]);

            lossSum += loss;
            count++;
        }

        double averageLoss = lossSum / count;
        return (float)Math.Exp(averageLoss);
    }
}