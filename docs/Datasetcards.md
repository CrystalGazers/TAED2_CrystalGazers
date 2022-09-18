---
annotations_creators:
- no-annotation
language:
- ca
language_creators:
- crowdsourced
license:
- cc-by-sa-3.0
multilinguality:
- monolingual
pretty_name: "Viquip\xE8dia"
size_categories:
- unknown
source_datasets:
- original
tags:
- wikipedia
- catalan
- text
- articles
- educational
task_categories:
- text-generation
task_ids:
- language-modeling
---

## Dataset Description

- **Link:** [https://www.kaggle.com/datasets/jarfo1/viquipdia]()
- **Main author:** [José Andrés Rodriguez Fonollosa](https://www.kaggle.com/jarfo1)

### Dataset summary

The Wikipedia dataset is a collection of scraped Wikipedia pages. The dataset is defined in catalan language, thus the model will be trained to recognize input exclusively in catalan.

### Supported tasks

Text generation

### Languages

Catalan

## Dataset structure

```
{
  'ca-2': [
    'ca.wiki.test.tokens',
    'ca.wiki.train.tokens',
    'ca.wiki.valid.tokens']
  'ca-100': [
    'ca.wiki.test.tokens',
    'ca.wiki.train.tokens',
    'ca.wiki.valid.tokens']
  'ca-all': [
    'ca.wiki.test.tokens',
    'ca.wiki.train.tokens',
    'ca.wiki.valid.tokens']
}
```

### Data fields

Plain text

### Data splits

|                         | train  | validation | test |
|-------------------------|-------:|-----------:|-----:|
| ca-2                    |10.64MB |1.07MB      |1.06MB|
| ca-100                  |528.96MB|1.07MB      |1.06MB|
| ca-all                  |1.32GB  |1.07MB      |1.06MB|

## Dataset creation

The dataset has been created by scraping all articles in catalan from Wikipedia. The original text has been crowdsourced by thousands of anonymous volunteers.

## Annotations

The dataset has no annotations as the files only contain the plain text from the articles.

## Considerations for Using the Data

The content has not been reviewed nor filtered by the authors of the dataset. It is supposed to have been reviewed by the Wikipedia community of volunteers to guarantee its veracity, lack of sensible or private content, and neutrality with respect to possible biases. Nonetheless, there's no guarantee that all 100% of the text has surpassed this review, or an indication of what text has been reviewed or not.

