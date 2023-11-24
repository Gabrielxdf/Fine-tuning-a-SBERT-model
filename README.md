# Ajuste fino  utilizando o framework Sentence Transformers
#### Ajuste fino de um modelo Setence-BERT utilizado no site [Vagas Anápolis](https://vagas.bcc.ifg.edu.br/) para recomendação de currículos e vagas. Todo este ajuste-fino foi baseado no artigo [conSultantBERT: Fine-tuned Siamese Sentence-BERT for Matching Jobs and Job Seekers](https://arxiv.org/abs/2109.06501) e o código foi baseado nesse tutorial do [SBERT](https://www.sbert.net/docs/training/overview.html)

---
# Primeiro, vamos falar sobre os dados
Os dados são compostos em um arquivo CSV contendo os seguinte campos:

- curriculos: contendo o texto extraído de um currículo em PDF.
- vagas: contendo o texto das descrições das vagas de emprego.
- notas: contendo uma nota de 1 a 5 indicando o grau de relevância do currículo e vaga respectivo. 1 para nenhnuma relevância e 5 para muita relevância.
  
Todos os dados foram gerados artificialemten com a ajuda do Chat-GPT. As notas de relevância foram inseridas manualmente.

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
df_dados["notas"] = df_dados["notas"].apply(lambda x: x * 0.2)
# 1 -> 0.2
# 2 -> 0.4
# 3 -> 0.6
# 4 -> 0.8
# 5 -> 1.0
```
Aqui transformamos as notas de relevância para graus de similaridades, pois utilizaremos a função de perda ```CosineSimilarityLoss``` quer requer um valor de similaridade entre ```0.0``` e ```1.0```. \
Tínhamos duas opções para fazer esse mapeamento:

- Fazer da nota mínima ```0.0``` de similaridade e a nota máxima ```0.8``` de similaridade.
- Fazer da nota mínima ```0.2``` de similaridade e a nota máxima ```1.0``` de similaridade.

Optou-se pela segunda opção pois é mais importante que o modelo aprenda mais sobre os pares currículo-vaga relevantes (atribui a maior similaridade ```1.0``` para os pares de muita relevância) do que aprender mais sobre os pares não-relevantes (atribui a menor similaridade ```0.0``` para os pares de nenhuma relevância). Em outras palavras, é mais importante que o modelo saiba recomendar bem pares currículo-vaga relevantes do que filtrar os pares que não são relevantes.

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
# Finalmente, o treinamento
```python
checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

word_embedding_model = models.Transformer(checkpoint, cache_dir=f'model/{checkpoint}')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```
Seguindo o tutorial do [SBERT](https://www.sbert.net/docs/training/overview.html), estamos criando um modelo SBERT manualmente com uma camada de embedding e uma camada de pooling CLS. O modelo utilizado na camdada de embedding neste exemplo é o ```sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2```, por padrão, ele gera um embedding de 384 dimensões, para mais informações sobre este modelo consulte o [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).\
Para alterar o método de pooling, basta alterar o parâmetro ```pooling_mode``` para ```'mean'```, por exemplo.

```python
train_loss = losses.CosineSimilarityLoss(model)
```
Utilizamos a função de perda ```CosineSimilarityLoss``` que você pode consultar mais informações [aqui](https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss).\
Em resumo, para cada par currículo-vaga será gerado seus respectivos embeddings e então a similaridade de cosseno desses embeddings é calculada. O resultado é então comparado com uma similaridade de cosseno referência para aquele par currículo-vaga, que no nosso caso, foi gerado manualmente.

```python
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='sbert')
```
Queremos medir a performance do modelo ao longo do treinamento, para isso vamos utilizar o ```EmbeddingSimilarityEvaluator.from_input_examples()``` com os dados de validação que consiste de ```InputExample```. Essa validação é executada periodicamente durante o treinamento. Além disso, essa validação retorna um score a cada execução e apenas o modelo com o score mais alto será salvo.

```python
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, evaluator=evaluator, show_progress_bar=True, output_path=f'model_FT/{checkpoint}')
```
Realizamos, enfim, o treinamento. Ajustamos o modelo chamando o método ```model.fit()```. Passamos uma lista de ```train_objectives```, os nossos objetivos de treinamento, que consiste em uma tupla ```(dataloader, loss_function)```. Também passamos nosso método de validação, juntamente com ```show_progress_bar=True``` para que seja exibida uma barra de progresso durante o processamento e  ```output_path``` para indicar o caminho onde será salvo o melhor modelo.

---
# Informações técnicas
## Evaluation Results
Os resultados das validações estão disponíveis neste [link](https://drive.google.com/file/d/1tH7_3WaKyBvzQb5nUmM8NfTA2kneAad6/view?usp=sharing) em um arquivo CSV.

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
