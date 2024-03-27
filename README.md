[![pt-br](https://img.shields.io/badge/lang-pt--br-green.svg)](README.pt-br.md)
# Fine-tuning using the Sentence Transformers framework
#### Fine-tuning a Setence-BERT model used on the [Vagas Anápolis](https://vagas.bcc.ifg.edu.br/) website to recommend CVs and jobs. All this fine-tuning was based on the [conSultantBERT: Fine-tuned Siamese Sentence-BERT for Matching Jobs and Job Seekers](https://arxiv.org/abs/2109.06501) article and the code was based on this tutorial from [SBERT](https://www.sbert.net/docs/training/overview.html)

---
# Google Colab
You can run this code using the free GPU offered by Google through [link](https://colab.research.google.com/github/Gabrielxdf/MachineLearning/blob/main/SBERT_FINE_TUNNING.ipynb).

---
# First, let's talk about the data
The data is all in portuguese composed in a CSV file containing the following fields:

- curriculos (CVs): Each line in this attribute refers to all text extracted from a CV in PDF format.
- vagas (jobs): each line in this attribute contains text that describes the job opening. As a job can be associated with several CVs, this content can be repeated for different CVs.
- notas (scores): containing the degree of relevance between the resume and the job description. 1 being low relevance, and 5 being high relevance. This annotation was made manually by students from the Vagas Anápolis project.
  

```python
def get_data_csv():
    df_data = pd.read_csv('data.csv')
    df_data.to_csv('data.csv', index=False, encoding='utf-8')
    return df_data

df_data = get_data_csv()
```
We load the data from the [data.csv](src/data.csv) file and place it in a Pandas DataFrame to better handle it.

```python
df_data["cvs"] = df_data["curriculos"].apply(lambda x: re.sub('\d+', '', x))
df_data["jobs"] = df_data["vagas"].apply(lambda x: re.sub('\d+', '', x))
```
We remove digits from cvs and jobs.

```python
df_data["scores"] = min_max_scaler.fit_transform(df_data["notas"].values.reshape(-1, 1))
# 1 -> 0.00
# 2 -> 0.25
# 3 -> 0.50
# 4 -> 0.75
# 5 -> 1.00
```
To fit the CosineSimilarityLoss loss function, we normalized the relevance scores using the MinMax strategy. Thus, the maximum scores receive 1.0 similarity, and the minimum scores 0.0.

---
# Now, let's prepare the data for training

```python
data_examples = []
for index, row in df_data.iterrows():
    data_examples.append(InputExample(texts=[row['cvs'], row['jobs']], label=row['score']))
```
We create a ```data_examples``` list that will contain an ```InputExample``` for each row of our Pandas DataFrame. ```InputExample``` is a data structure used for fine-tuning SBERT. In our case, it will contain a cv-job text pair and their respective similarity.

```python
data_examples = shuffle(data_examples, random_state=42)
train_index = int(len(data_examples) * 0.8)
val_index = int(len(data_examples) * 0.2)

train_examples = data_examples[:train_index]
val_examples = data_examples[train_index:train_index+val_index]
test_examples = data_examples[train_index+val_index:]
```
We are splitting the data into 60% for training, 20% for validation and 20% for tests.

```python
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
```
Creating a PyTorch DataLoader with the training data, so we can iterate over the data in batches with the parameter ```batch_size=4```.

---
# The fine-tuning
```python
checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

word_embedding_model = models.Transformer(checkpoint, cache_dir=f'model/{checkpoint}')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```
Following the [SBERT](https://www.sbert.net/docs/training/overview.html) tutorial, we are loading a pre-trained SBERT model, with embedding layers and a CLS pooling layer. The model used in the embedding layer in this example is ```sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2```. By default, it generates a 384-dimensional embedding, for more information about this model see the [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).\
To change the pooling method, simply change the ```pooling_mode``` parameter to ```'mean'```, for example.

```python
train_loss = losses.CosineSimilarityLoss(model)
```

For fine-tuning training, we use the ```CosineSimilarityLoss``` loss function, access [aqui](https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss) for more details.\
In short, for each cv-job its respective embeddings will be generated and then the cosine similarity score of these embeddings is used to correct the model weights. The result is then compared with a reference cosine similarity score for that cv-job pair.

```python
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='sbert')
```
We want to measure the model's performance throughout training, for this we will use ```EmbeddingSimilarityEvaluator.from_input_examples()``` with the validation data consisting of ```InputExample```. This validation is performed iteratively during training. Furthermore, this validation returns a score every run and only the model with the highest score will be saved.

```python
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, evaluator=evaluator, show_progress_bar=True, output_path=f'model_FT/{checkpoint}')
```
Finally, we carry out the training. We adjust the model by calling the ```model.fit()``` method. We pass a list of ```train_objectives```, which consists of a tuple ```(dataloader, loss_function)```. We also pass our validation method, along with ```show_progress_bar=True``` so that a progress bar is displayed during processing and ```output_path``` to indicate the path where the best model will be saved.

---
# Let's test!
Let's test our fine-tuning with a simple prototype of a job recommendation system for a resume. Remember, all of our data is in portuguese.

```python
cv_test = 'Nome: Laura Costa - Objetivo: Busco uma posição como Analista Econômico, onde posso aplicar minha formação acadêmica em Economia e aprimorar minhas habilidades em análise econômica. Formação Acadêmica: Bacharelado em Economia - Universidade Federal de Estado Y (-) Experiência Profissional: Assistente de Análise Econômica - Empresa de Consultoria Econômica LTDA - Cidade Financeira, Estado Y (-Presente) Coleta de dados econômicos. Auxílio na elaboração de relatórios e análises. Habilidades: Conhecimentos intermediários em análise econômica. Familiaridade com ferramentas como Excel e SPSS. Idiomas: Inglês: Avançado Espanhol: Básico'
```
Let's use this resume that was taken from our database for the test. It is the CV of a person with a degree in **Economics** seeking a position as an **Economic Analyst**. The objective is to recommend the most similar jobs for this CV, in descending order.

```python
jobs_test = list(set([test_example.texts[1] for test_example in test_examples]))
```
Here we are creating a list of all the jobs in the test set. The set() function is to create a set, thus eliminating duplicate records. list() is to transform the set into a list again, since the set cannot be accessed by index, something that will be important to us later.

```python
cv_embedding = model.encode(cv_test)
jobs_embedding = [model.encode(vaga) for vaga in jobs_test]
```
Obtaining the embedding of the chosen resume and all jobs in our test set. The function that does this is model.encode().

```python
similarity_score = util.cos_sim(cv_embedding, jobs_embedding)
```
In this line of code, the similarity *score* of all cv-job pairs is obtained, considering their embeddings.

```python
pairs = []
for index, score in enumerate(similarity_score[0]):
    pairs.append({"index": index, "score": score})
```
Just adding an index for each similarity score. This index will indicate which job this *score* is about. This will make it easier to retrieve the texts of jobs after sorting the similarity scores in descending order.

```python
pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)
```
Order the similarity *scores* of cv-job pairs from highest to lowest.

```python
print(f' CV: {cv_test} \n\n')
for pair in pairs[0:5]:
    print(f' Job: {jobs_test[pair["index"]]} \n Predicted similarity score after fine-tuning: {pair["score"]} \n')
```
Finally, we are just displaying the CV and its 5 most relevant jobs, along with the *similarity score* obtained.

---
# Technical information
## Evaluation Results
Validation results are available at this [link](https://drive.google.com/file/d/1FrYwcDT3jFTBsaEdcI9BSVSaBYFvKcNL/view?usp=sharing) in a CSV file.

## Training
The model was trained with the parameters:

**DataLoader**:

`torch.utils.data.dataloader.DataLoader` of length 1240 with parameters:
```
{'batch_size': 4, 'sampler': 'torch.utils.data.sampler.RandomSampler', 'batch_sampler': 'torch.utils.data.sampler.BatchSampler'}
```

**Loss**:

`sentence_transformers.losses.CosineSimilarityLoss.CosineSimilarityLoss` 

Parameters of the fit()-Method:
```
{
    "epochs": 5,
    "evaluation_steps": 0,
    "evaluator": "sentence_transformers.evaluation.EmbeddingSimilarityEvaluator.EmbeddingSimilarityEvaluator",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'torch.optim.adamw.AdamW'>",
    "optimizer_params": {
        "lr": 2e-05
    },
    "scheduler": "WarmupLinear",
    "steps_per_epoch": null,
    "warmup_steps": 10000,
    "weight_decay": 0.01
}
```


## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

Para maiores informações acesse:
- https://www.sbert.net/docs/training/overview.html
- https://www.sbert.net/docs/package_reference/evaluation.html
- https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss
- https://www.sbert.net/docs/package_reference/models.html
- https://www.sbert.net/docs/usage/semantic_textual_similarity.html
