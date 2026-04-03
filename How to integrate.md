# How to integrate for stage 3
## Як підключити Corpus – Tokenizer – NGram у Trainer CLI?
Для використання повного пайплайну Corpus – Tokenizer – NGram необхідно зробити наступні кроки:
1. Завантажити корпус
```csharp
CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
Corpus corpus = loader.Load("../../../../../data/showcase.txt", new Lib.Corpus.Configuration.CorpusLoadOptions { Lowercase = true });
```
2. Завантажити корпус
```csharp
WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
int[] tokens = tokenizer.Encode(corpus.TrainText);
```
3. Завантажити корпус
```csharp
TrigramModel model = new TrigramModel(tokenizer.VocabSize);
model.Train(tokens);
```
4. Завантажити корпус
```csharp
List<int> context = new List<int>
{
    tokens[2],
    tokens[3]
};

for (int i = 0; i < 10; i++)
{
    float[] scores = model.NextTokenScores(context.ToArray());
    context.Add(Array.IndexOf(scores, scores.Max()) + 1);
}
```
5. Завантажити корпус
```csharp
string text = tokenizer.Decode(context.ToArray());
```

Роль кожного компонента
|Компонент|Роль|
|-----|-----|
|Corpus|dddd|
|Tokenizer|dddd|
|NGram/Trigram|dddd|

## Формат checkpoint для bigram та trigram

## Як побудувати greedy generation loop для Chat?
Greedy generation - 
Код
Як це працює
Беремо початкові токени
Модель дає ймовірності
Беремо найбільш ймовірний токен
Додаємо його до контексту
Повторюємо
Особливості
Bigram → дивиться на 1 токен
Trigram → дивиться на 2 токени
fallback → рівномірний розподіл

## word vs char – яку гранулярнiсть обрати
*Детальніше можна прочитати в README.md "Word vs Char"*
WordTokenizer

Працює з цілими словами

Плюси:
зрозумілий текст
швидке навчання
менше кроків генерації
Мінуси:
є <UNK> (невідомі слова)
великий словник

CharTokenizer

Працює з символами

Плюси:
немає <UNK>
може генерувати нові слова
малий словник
Мінуси:
гірша читабельність
довша генерація
складніше навчання

 Висновок

Для baseline (B4):


 WordTokenizer — кращий вибір, бо:

генерує зрозумілі українські слова
простіший для інтеграції


CharTokenizer варто використовувати:

якщо важлива гнучкість
або для експериментів