using MiniChatGPT.Contracts;

public class NGramModelFactory : INGramModelFactory
{
    public ILanguageModel Create(string modelType, int vocabSize)
    {
        if (string.IsNullOrWhiteSpace(modelType))
        {
            throw new ArgumentException("Model type cannot be empty.", nameof(modelType));
        }

        if (modelType.Equals("bigram", StringComparison.OrdinalIgnoreCase))
        {
            return new NGramModel(vocabSize);
        }

        if (modelType.Equals("trigram", StringComparison.OrdinalIgnoreCase))
        {
            return new TrigramModel(vocabSize);
        }

        throw new ArgumentException($"Unknown model type: {modelType}");
    }
}