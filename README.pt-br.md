[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
# Ajuste fino  utilizando o framework Sentence Transformers
#### Ajuste fino de um modelo Setence-BERT utilizado no site [Vagas Anápolis](https://vagas.bcc.ifg.edu.br/) para recomendação de currículos e vagas. Todo este ajuste fino foi baseado no artigo [conSultantBERT: Fine-tuned Siamese Sentence-BERT for Matching Jobs and Job Seekers](https://arxiv.org/abs/2109.06501) e o código foi baseado nesse tutorial do [SBERT](https://www.sbert.net/docs/training/overview.html)

---
# Google Colab
Você pode executar esse código utilizando a GPU gratuita ofertada pelo Google através do [link](https://colab.research.google.com/github/Gabrielxdf/MachineLearning/blob/main/SBERT_FINE_TUNNING.ipynb).

---
# Primeiro, vamos falar sobre os dados
Os dados são compostos em um arquivo CSV contendo os seguinte campos:

- curriculos: cada linha neste atributo se refere a todo o texto extraído de um currículo em formato PDF.
- vagas: cada linha neste atributo contém o texto que descreve a vaga de emprego. Como uma vaga pode estar associada a vários currículos, esse conteúdo pode se repetir para distintos currículos.
- notas: contendo o grau de relevância entre o currículo e a descrição da vaga. Sendo 1 para baixa relevância, e 5 para alta relevância. Essa anotação foi feita manualmente pelos alunos do projeto Vagas Anápolis.
  

```python
def get_data_csv():
    df_data = pd.read_csv('data.csv')
    df_data.to_csv('data.csv', index=False, encoding='utf-8')
    return df_data

df_data = get_data_csv()
```
Carregamos os dados do arquivo [data.csv](src/data.csv) e colocamos em um DataFrame do Pandas para melhor manuseá-lo.

```python
df_data["cvs"] = df_data["curriculos"].apply(lambda x: re.sub('\d+', '', x))
df_data["jobs"] = df_data["vagas"].apply(lambda x: re.sub('\d+', '', x))
```
Removemos os dígitos dos currículos e das vagas.

```python
df_data["score"] = min_max_scaler.fit_transform(df_data["notas"].values.reshape(-1, 1))
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
for index, row in df_data.iterrows():
    data_examples.append(InputExample(texts=[row['csv'], row['jobs']], label=row['score']))
```
Criamos uma lista ```data_examples``` que irá conter um ```InputExample``` para cada linha do nosso DataFrame Pandas. O ```InputExample``` é uma estrutura de dados usada para o ajuste fino do SBERT. No nosso caso ela irá conter um par de texto currículo-vaga e sua respectiva similaridade.

```python
data_examples = shuffle(data_examples, random_state=42)
train_index = int(len(data_examples) * 0.8)
val_index = int(len(data_examples) * 0.2)

train_examples = data_examples[:train_index]
val_examples = data_examples[train_index:train_index+val_index]
test_examples = data_examples[train_index+val_index:]
```
Estamos dividindo os dados em 60% para treino, 20% para validação e 20% para testes.

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
Seguindo o tutorial do [SBERT](https://www.sbert.net/docs/training/overview.html), estamos carregando um modelo SBERT pré-treinado, com as camadas de embedding e uma camada de pooling CLS. O modelo utilizado na camdada de embedding neste exemplo é o ```sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2```. Por padrão, ele gera um embedding de 384 dimensões, para mais informações sobre este modelo consulte o [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).\
Para alterar o método de pooling, basta alterar o parâmetro ```pooling_mode``` para ```'mean'```, por exemplo.

```python
train_loss = losses.CosineSimilarityLoss(model)
```

Para o treinamento no ajuste fino, utilizamos a função de perda ```CosineSimilarityLoss```, acesse [aqui](https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss) para mais detalhes.\
Em resumo, para cada currículo-vaga será gerado seus respectivos embeddings e então a similaridade de cosseno desses embeddings é utilizada para correção dos pesos do modelo. O resultado é então comparado com uma similaridade de cosseno referência para aquele par currículo-vaga.

```python
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='sbert')
```
Queremos medir a performance do modelo ao longo do treinamento, para isso vamos utilizar o ```EmbeddingSimilarityEvaluator.from_input_examples()``` com os dados de validação que consiste de ```InputExample```. Essa validação é executada iterativamente durante o treinamento. Além disso, essa validação retorna um score a cada execução e apenas o modelo com o score mais alto será salvo.

```python
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, evaluator=evaluator, show_progress_bar=True, output_path=f'model_FT/{checkpoint}')
```
Realizamos, enfim, o treinamento. Ajustamos o modelo chamando o método ```model.fit()```. Passamos uma lista de ```train_objectives```, os nossos objetivos de treinamento, que consiste em uma tupla ```(dataloader, loss_function)```. Também passamos nosso método de validação, juntamente com ```show_progress_bar=True``` para que seja exibida uma barra de progresso durante o processamento e  ```output_path``` para indicar o caminho onde será salvo o melhor modelo.

---
# Vamos testar!
Vamos testar nosso ajuste fino com um simples protótipo de um sistema de recomendação de vagas para um currículo.

```python
cv_test = 'Nome: Laura Costa - Objetivo: Busco uma posição como Analista Econômico, onde posso aplicar minha formação acadêmica em Economia e aprimorar minhas habilidades em análise econômica. Formação Acadêmica: Bacharelado em Economia - Universidade Federal de Estado Y (-) Experiência Profissional: Assistente de Análise Econômica - Empresa de Consultoria Econômica LTDA - Cidade Financeira, Estado Y (-Presente) Coleta de dados econômicos. Auxílio na elaboração de relatórios e análises. Habilidades: Conhecimentos intermediários em análise econômica. Familiaridade com ferramentas como Excel e SPSS. Idiomas: Inglês: Avançado Espanhol: Básico'
```
Vamos usar esse currículo que foi tirado da nossa base de dados para o teste. É o currículo de uma pessoa formada em **Economia** buscando uma vaga de **Analista Econômico**. O objetivo é recomendar as vagas mais similares para esse currículo, em ordem decrescente.

```python
jobs_test = list(set([test_example.texts[1] for test_example in test_examples]))
```
Aqui estamos criando uma lista com todas as vagas do conjunto de teste. A função set() é para criar um conjunto, assim eliminando os registros duplicados. A list() é para transformar o conjunto em uma lista novamente, visto que o conjunto não pode ser acessado por índice, coisa que nos será importante posteriormente.

```python
cv_embedding = model.encode(cv_test)
jobs_embedding = [model.encode(vaga) for vaga in jobs_test]
```
Obtendo o embedding do currículo escolhido e de todas as vagas do nosso conjunto de teste. A função que faz isso é a model.encode().

```python
similarity_score = util.cos_sim(cv_embedding, jobs_embedding)
```
Nesta linha de código, obtém-se o *score* de similaridade de todos os pairs currículo-vaga, considerando seus embeddings.

```python
pairs = []
for index, score in enumerate(similarity_score[0]):
    pairs.append({"index": index, "score": score})
```
Apenas adicionando um índice para cada similaridade. Esse índice indicará de qual vaga esse *score* se trata. Isso facilitará na hora de recuperar os textos das vagas após a ordenação decrescente das similaridades.

```python
pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)
```
Ordena os *scores* de similaridade dos pairs currículo-vaga do maior para o menor.

```python
print(f' CV: {cv_test} \n\n')
for pair in pairs[0:5]:
    print(f' Job: {jobs_test[pair["index"]]} \n Predicted similarity score after fine-tuning: {pair["score"]} \n')
```
Por fim, estamos apenas exibindo o currículo e suas 5 vagas mais relevantes, juntamente com o *score* de similaridade obtido.

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
- https://www.sbert.net/docs/usage/semantic_textual_similarity.html
