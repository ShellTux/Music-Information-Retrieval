#import "@preview/ilm:1.4.1": *

#set text(lang: "pt")

#let authors = (
  "Luís Pedro de Sousa Oliveira Góis, nº 2018280716",
  "Gonçalo José dos Santos Silva, nº 2022233004",
  "Pedro Francisco Madureira Garcia Teixeira, nº 2017261525",
  "Renato Marques Reis, nº 2022232936",
).sorted().join("\n")

#let today = datetime(day: 10, month: 05, year: 2025)

#show: ilm.with(
  title: "Relatório de Multimédia",
  author: authors,
  date: today,
  bibliography: bibliography("references.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

= Music Information Retrieval

// == Introdução
//
// // TODO: explicar music information retrieval e dataset utilizado

== 3.5. Apresentar, comparar e discutir os resultados.

Rankings (top 10 + query) guardados em ficheiros ranking\_\*.csv

Top 10 Recomendações para MT0000414517.mp3 (excluindo a própria música):

Recomendações - Distância Euclidiana:

#table(
  columns: 3,
  table.header[][*Filename*][*Distância*],
  [1], [MT0003949060.mp3], [2.0477],
  [2], [MT0001515531.mp3], [2.1022],
  [3], [MT0004274911.mp3], [2.1153],
  [4], [MT0009897495.mp3], [2.1383],
  [5], [MT0000040632.mp3], [2.1396],
  [6], [MT0007043936.mp3], [2.1503],
  [7], [MT0003900455.mp3], [2.1508],
  [8], [MT0005469880.mp3], [2.1596],
  [9], [MT0004032071.mp3], [2.1635],
  [10], [MT0034005433.mp3], [2.2113],
)

Recomendações - Distância Manhattan:
#table(
  columns: 3,
  table.header[][*Filename*][*Distância*],
  [1], [MT0003949060.mp3], [15.6174],
  [2], [MT0004274911.mp3], [17.4131],
  [3], [MT0000040632.mp3], [17.5722],
  [4], [MT0005469880.mp3], [18.0741],
  [5], [MT0003900455.mp3], [18.0762],
  [6], [MT0000218346.mp3], [18.0827],
  [7], [MT0008401073.mp3], [18.2530],
  [8], [MT0001515531.mp3], [18.2831],
  [9], [MT0009897495.mp3], [18.4433],
  [10], [MT0001624303.mp3], [18.6606],
)

Recomendações - Distância Cosseno:
#table(
  columns: 3,
  table.header[][*Filename*][*Distância*],
  [1], [MT0003949060.mp3], [0.0575],
  [2], [MT0004274911.mp3], [0.0592],
  [3], [MT0001515531.mp3], [0.0605],
  [4], [MT0009897495.mp3], [0.0622],
  [5], [MT0000040632.mp3], [0.0626],
  [6], [MT0003900455.mp3], [0.0629],
  [7], [MT0004032071.mp3], [0.0632],
  [8], [MT0002634024.mp3], [0.0633],
  [9], [MT0007043936.mp3], [0.0638],
  [10], [MT0005469880.mp3], [0.0644],
)

Na Secção 3 foram calculados os rankings de similaridade utilizando as
métricas Euclidiana, Manhattan e de Cosseno. Verificou-se que existe
uma grande sobreposição nas recomendações entre os rankings, com
músicas como MT0003949060.mp3, MT0004274911.mp3 e MT0000040632.mp3 a
surgirem consistentemente nos três rankings.

A métrica de Cosseno destacou-se pela proximidade muito pequena entre a
música de consulta e as restantes recomendações, sugerindo elevada
discriminação vetorial. A Euclidiana apresentou valores maiores, como
esperado devido à sua natureza baseada em distância absoluta. A
Manhattan teve as maiores distâncias, mas também incluiu recomendações
semelhantes. No entanto, introduziu ligeiras variações, como a inclusão
de músicas como MT0000218346.mp3 e MT0001624303.mp3, que não surgiram
nos outros rankings, refletindo uma maior sensibilidade a diferenças
acumuladas nos vetores.

Em suma, as três métricas demonstraram coerência geral nas
recomendações, com o Cosseno a revelar-se o mais "apertado" em termos
de variação das distâncias.

== 4.1.4. Apresentar, comparar e discutir os resultados.

```
Ranking: Euclidean-------------
['MT0000414517.mp3', 'MT0003949060.mp3', 'MT0001515531.mp3',
'MT0004274911.mp3', 'MT0009897495.mp3', 'MT0000040632.mp3',
'MT0007043936.mp3', 'MT0003900455.mp3', 'MT0005469880.mp3',
'MT0004032071.mp3', 'MT0034005433.mp3']
[0. 2.04773879 2.10221685 2.1152786 2.13828185 2.13955751
2.15032851 2.15084032 2.15959407 2.16352387 2.21132133]

Ranking: Manhattan-------------
['MT0000414517.mp3', 'MT0003949060.mp3', 'MT0004274911.mp3',
'MT0000040632.mp3', 'MT0005469880.mp3', 'MT0003900455.mp3',
'MT0000218346.mp3', 'MT0008401073.mp3', 'MT0001515531.mp3',
'MT0009897495.mp3', 'MT0001624303.mp3']
[ 0. 15.61736689 17.41313885 17.57218606 18.07409526 18.07618288
18.08268368 18.25300309 18.28311896 18.44327003 18.66056736]

Ranking: Cosine-------------
['MT0000414517.mp3', 'MT0003949060.mp3', 'MT0004274911.mp3',
'MT0001515531.mp3', 'MT0009897495.mp3', 'MT0000040632.mp3',
'MT0003900455.mp3', 'MT0004032071.mp3', 'MT0002634024.mp3',
'MT0007043936.mp3', 'MT0005469880.mp3']
[0. 0.05749657 0.05920408 0.06054802 0.06219172 0.06262757
0.06292372 0.06324029 0.06326967 0.06379077 0.06435431]

Ranking: Metadata-------------
['MT0000414517.mp3', 'MT0027048677.mp3', 'MT0010489498.mp3',
'MT0010487769.mp3', 'MT0033397838.mp3', 'MT0003949060.mp3',
'MT0012331779.mp3', 'MT0000040632.mp3', 'MT0002222957.mp3',
'MT0008222676.mp3', 'MT0010900969.mp3']
[7 7 7 7 7 7 7 7 7 3 3]

Precision de: 20.0
Precision dm: 20.0
Precision dc: 20.0
```

A avaliação objetiva foi feita com base nos metadados, comparando
artista, género e estado de espírito. Todas as métricas de distância
apresentaram uma precisão de 20% (2 recomendações em 10 coincidiram com
o ranking de metadados).

Isto mostra que, apesar das métricas baseadas em conteúdo conseguirem
capturar certas semelhanças acústicas, a correspondência com a
informação semântica (metadados) ainda é limitada.

É notável que a música MT0003949060.mp3 surgiu tanto nos rankings de
conteúdo como no de metadados, indicando que pode ter uma representação
consistente entre o conteúdo áudio e os metadados. No entanto, outras
músicas relevantes do ponto de vista contextual (como MT0027048677.mp3)
não foram captadas pelas métricas de conteúdo.

Este resultado confirma a importância de complementar métricas baseadas
em conteúdo com informação contextual para melhores sistemas de
recomendação.

== 4.2.3. Apresentar, comparar e discutir os resultados.

A avaliação subjetiva revelou uma aceitação elevada das recomendações. A métrica Euclidiana obteve a
média global mais elevada (3.83), seguida de Metadados (3.67), Cosseno (3.60) e Manhattan (3.42). Todas
as métricas atingiram 100% de precisão subjetiva (todas as recomendações com média >= 2.5).

O desvio-padrão mais baixo foi registado pela Euclidiana (0.33), sugerindo maior consistência nas
avaliações entre membros. A Manhattan teve maior dispersão (0.59), refletindo maior variabilidade na
perceção da qualidade das suas recomendações.

As recomendações baseadas em metadados foram bem avaliadas, mas as métricas baseadas em conteúdo
rivalizaram eficazmente, especialmente a Euclidiana e o Cosseno.

Este resultado reforça que a abordagem baseada em conteúdo, quando bem executada, pode gerar
recomendações com alta aceitação pelos utilizadores, mesmo sem usar metadados.

// == Preparação
//
// === Analisar a base de dados
//
// == Extração de features
//
// == Implementação de métricas de similaridade
//
// == Avaliação
//
// === Avaliação objetiva
//
// === Avaliação subjetiva
