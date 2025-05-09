#import "@preview/ilm:1.4.1": *

#set text(lang: "pt")

#let authors = (
  "Luís Pedro de Sousa Oliveira Góis, nº 2018280716",
  "Gonçalo José dos Santos Silva, nº 2022233004",
  "Pedro Francisco Madureira Garcia Teixeira, nº 2017261525",
  "Renato Marques Reis, nº 2022232936",
).sorted().join("\n")

#show: ilm.with(
  title: "Relatório de Multimédia",
  author: authors,
  date: datetime.today(),
  bibliography: bibliography("references.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

= Music Information Retrieval

== Introdução

// TODO: explicar music information retrieval e dataset utilizado

== Preparação

=== Analisar a base de dados

== Extração de features

== Implementação de métricas de similaridade

== Avaliação

=== Avaliação objetiva

=== Avaliação subjetiva
