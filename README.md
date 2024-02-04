# Ajuste fino  utilizando o framework Sentence Transformers
#### Ajuste fino de um modelo Setence-BERT utilizado no site [Vagas Anápolis](https://vagas.bcc.ifg.edu.br/) para recomendação de currículos e vagas. Todo este ajuste-fino foi baseado no artigo [conSultantBERT: Fine-tuned Siamese Sentence-BERT for Matching Jobs and Job Seekers](https://arxiv.org/abs/2109.06501) e o código foi baseado nesse tutorial do [SBERT](https://www.sbert.net/docs/training/overview.html)

---
# Primeiro, vamos falar sobre os dados
Os dados são compostos em um arquivo CSV contendo os seguinte campos:

- curriculos: cada linha neste atributo se refere a todo o texto extraído de um currículo em formato PDF.
- vagas: cada linha neste atributo contém o texto que descreve a vaga de emprego. Como uma vaga pode estar associada a vários currículos, esse conteúdo pode se repetir para distintos currículos.
- notas: contendo o grau de relevância entre o currículo e a descrição da vaga. Sendo 1 para baixa relevância, e 5 para alta relevância. Essa anotação foi feita manualmente pelos alunos do projeto Vagas Anápolis.
  

```python
def obter_dados_csv():
    df_dados = pd.read_csv('dados.csv')
    df_dados.to_csv('dados.csv', index=False, encoding='utf-8')
    return df_dados

df_dados = obter_dados_csv()
```
Carregamos os dados do arquivo [dados.csv](dados.csv) e colocamos em um DataFrame do Pandas para melhor manuseá-lo.

```python
df_dados["curriculos"] = df_dados["curriculos"].apply(lambda x: re.sub('\d+', '', x))
df_dados["vagas"] = df_dados["vagas"].apply(lambda x: re.sub('\d+', '', x))
```
Removemos os dígitos dos currículos e das vagas.

```python
df_dados["notas"] = min_max_scaler.fit_transform(df_dados["notas"].values.reshape(-1, 1))
# 1 -> 0.00
# 2 -> 0.25
# 3 -> 0.50
# 4 -> 0.75
# 5 -> 1.00
```
Para ajustar à função de perda CosineSimilarityLoss, fizemos normalização das notas de relevância usando a estratégia MinMax. Assim as notas máximas recebem 1.0 de similaridade, e as mínimas 0.0.

---
# Agora, vamos preparar os dados para o treinamento

```python
data_examples = []
for index, row in df_dados.iterrows():
    data_examples.append(InputExample(texts=[row['curriculos'], row['vagas']], label=row['notas']))
```
Criamos uma lista ```data_examples``` que irá conter um ```InputExample``` para cada linha do nosso DataFrame Pandas. O ```InputExample``` é uma estrutura de dados usada para o ajuste fino do SBERT. No nosso caso ela irá conter um par de texto currículo-vaga e sua respectiva similaridade.

```python
data_examples = shuffle(data_examples, random_state=42)
indice_treino = int(len(data_examples) * 0.8)

train_examples = data_examples[:indice_treino]
val_examples = data_examples[indice_treino:]
```
Estamos dividindo os dados em 80% para treino e 20% para validação.

```python
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
```
Criando um DataLoader do PyTorch com os dados de treinamento, para que possamos iterar sobre os dados em lotes com o parâmetro ```batch_size=4```.

---
# O ajuste fino
```python
checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

word_embedding_model = models.Transformer(checkpoint, cache_dir=f'model/{checkpoint}')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```
Seguindo o tutorial do [SBERT](https://www.sbert.net/docs/training/overview.html), estamos carregando um modelo SBERT pré-treinado. Com as camadas de embedding e uma camada de pooling CLS. O modelo utilizado na camdada de embedding neste exemplo é o ```sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2```, por padrão, ele gera um embedding de 384 dimensões, para mais informações sobre este modelo consulte o [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).\
Para alterar o método de pooling, basta alterar o parâmetro ```pooling_mode``` para ```'mean'```, por exemplo.

```python
train_loss = losses.CosineSimilarityLoss(model)
```

Para o treinamento no ajuste fino, utilizamos a função de perda ```CosineSimilarityLoss```, acesse [aqui](https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss) para mais detalhes.\
Em resumo, para cada currículo-vaga será gerado seus respectivos embeddings e então a similaridade de cosseno desses embeddings é utilizada para correção dos pesos. O resultado é então comparado com uma similaridade de cosseno referência para aquele par currículo-vaga.

```python
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='sbert')
```
Queremos medir a performance do modelo ao longo do treinamento, para isso vamos utilizar o ```EmbeddingSimilarityEvaluator.from_input_examples()``` com os dados de validação que consiste de ```InputExample```. Essa validação é executada iterativamente durante o treinamento. Além disso, essa validação retorna um score a cada execução e apenas o modelo com o score mais alto será salvo.

```python
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, evaluator=evaluator, show_progress_bar=True, output_path=f'model_FT/{checkpoint}')
```
Realizamos, enfim, o treinamento. Ajustamos o modelo chamando o método ```model.fit()```. Passamos uma lista de ```train_objectives```, os nossos objetivos de treinamento, que consiste em uma tupla ```(dataloader, loss_function)```. Também passamos nosso método de validação, juntamente com ```show_progress_bar=True``` para que seja exibida uma barra de progresso durante o processamento e  ```output_path``` para indicar o caminho onde será salvo o melhor modelo.

---
# Informações técnicas
## Evaluation Results
Os resultados das validações estão disponíveis neste [link](https://drive.google.com/file/d/1FrYwcDT3jFTBsaEdcI9BSVSaBYFvKcNL/view?usp=sharing) em um arquivo CSV.

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
