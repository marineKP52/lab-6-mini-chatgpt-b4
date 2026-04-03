# How to integrate for stage 3
## Як підключити Corpus – Tokenizer – NGram у Trainer CLI?
Для використання повного пайплайну Corpus – Tokenizer – NGram необхідно зробити наступні кроки:
1. Завантажити корпус
```csharp
CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
Corpus corpus = loader.Load("path to text file", new Lib.Corpus.Configuration.CorpusLoadOptions { Lowercase = true });
```
*Більше про конфігурацію корпуса можна знайти в README.md команди B1*

2. Сформувати токени
```csharp
WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
int[] tokens = tokenizer.Encode(corpus.TrainText);
```
3. Створити та натренувати модель
```csharp
TrigramModel model = new TrigramModel(tokenizer.VocabSize);
model.Train(tokens);
```
*Тут використано триграмну модель, але можна працювати і з біграмною (клас NGram)*

4. Згенерувати набір токенів "відповіді"
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
*Тут відповідь генерується наступним чином: беруться два початкові токена (це необхідно для триграмної моделі) та додаються до контекста, далі потрібну кількість ітерацій новий токен шукається шляхом пошуку токена з максимальною ймовірністю на базі нового контекста. Кожен згенерований токен додається до контекста.*

5. Перетворити токени на текст
```csharp
string text = tokenizer.Decode(context.ToArray());
```

Роль кожного компонента
|Компонент|Роль|
|-----|-----|
|Corpus|Формує навчальні та валідаційні дані з тексту|
|Tokenizer|Перетворює символи чи слова на токени та навпаки|
|NGram/Trigram|Мовна модель, навчається та генерує відповідь|

## Формат checkpoint для bigram та trigram
### Що таке Checkpoint

**Checkpoint** — це JSON-файл, який зберігає:

- тип моделі (**bigram / trigram**)
- токенізатор (**word / char**)
- словник токенів
- ваги моделі

Після завантаження checkpoint-а модель повинна генерувати **той самий результат**, що і до збереження.

### Структура Checkpoint

```json
{
  "ModelKind": "trigram",
  "TokenizerKind": "word",
  "TokenizerPayload": {
    "Words": [null, "калина", "похилилася", "журиться"]
  },
  "ModelPayload": {
    "modelKind": "trigram",
    "modelPayload": {
      "bigramProbs": [...],
      "trigramProbs": {
        "1, 2": [...]
      }
    }
  },
  "Seed": 42,
  "ContractFingerprintChain": "..."
}
```
### Bigram
- bigramProbs — матриця float[vocabSize][vocabSize]
- кожен рядок = ймовірності наступного токена
```
"bigramProbs": [
  [0.0, 0.5, 0.5],
  ...
]
```
### Trigram
- trigramProbs — словник
- ключ: "prev2, prev1" (рядок!)
- значення: масив ймовірностей
```
"trigramProbs": {
  "1, 2": [0.0, 0.0, 1.0],
  "2, 3": [0.0, 1.0, 0.0]
}
```
Ключ зберігається як рядок, бо (int,int) не підтримується в JSON.

### Важливі нюанси

- TokenizerPayload містить словник (Words або Chars)
- Words[0] = null → це <UNK>
- після Load() payload приходить як object, потрібно кастити до JsonElement
- trigram використовує bigram як fallback

### Гарантія коректності
Правильна реалізація checkpoint-а перевіряється так:
1. згенерувати текст
2. зберегти checkpoint
3. завантажти checkpoint
4. згенерувати ще раз
→ результати мають бути однакові

## Як побудувати greedy generation loop для Chat?
**Greedy generation loop** - відповідь моделі генеруєтся шляхом визначення найбільш ймовірного токену на базі контексту. Кожен згенерований символ додається до контексту.

**Початковий етап:**
Беремо навчену модель та додаємо до контексту один або два (для біграми та триграми відповідно) початкових токена.

**Цикл:**
1. Викликаємо NextTokenScores() для поточного контексту.
2. Серед ймовірностей шукаємо найбільшу.
3. Записуємо токен з цією ймовірність до контексту.

**Завершальний етап:** Переводимо токени в текст.

## word vs char – яку гранулярнiсть обрати
*Детальніше можна прочитати в README.md "Word vs Char"*

**WordTokenizer** - працює з цілими словами. Він генерує більш зрозумілий текст, навчається швидше та потребує менших кроків генерації для більшого об'єму тексту відповіді. Однак, він потребує словника більшого розміру.

**CharTokenizer** - працює з окремими символами. Він потребує словника меншого розміру та більш гнучкий, якщо треба формувати нові слова, однак через це може видавати менш зрозумілий текст через генерацію неіснуючих слів. Також сама генерація триває довше.

**Яку гранулярність обрати?** Якщо потрібен зрозуміліший текст та вища швидкість - WordTokenizer. Якщо потрібна гнучкість у формуванні слів - CharTokenizer.
