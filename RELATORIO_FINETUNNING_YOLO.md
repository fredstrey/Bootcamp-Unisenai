# Relatório de Fine-Tuning do YOLO para Detecção de Rachaduras

## Visão geral

Este projeto foi desenvolvido para detectar rachaduras em imagens usando um modelo da família YOLO treinado previamente e adaptado ao domínio específico do problema. A base do trabalho foi o arquivo `best.pt`, gerado a partir de um processo de fine-tuning sobre um conjunto de imagens rotuladas.

O objetivo principal não era construir um detector genérico do zero, mas aproveitar o conhecimento visual já aprendido por um modelo YOLO e especializá-lo para o cenário de rachaduras, que é um problema de visão computacional mais estreito e com dados mais limitados.

## Por que fine-tuning

A escolha por fine-tuning foi a estratégia mais segura e eficiente por alguns motivos:

- Foi disponibilizada uma quantidade razoável de imagens já rotuladas.
- Esse volume de dados é suficiente para adaptar um modelo pré-treinado, mas ainda é pequeno para justificar um treino completo do zero com boa robustez.
- Treinar do zero exigiria muito mais dados, mais tempo computacional e maior risco de overfitting ou de um modelo final pouco estável.
- O YOLO já carrega padrões visuais internos muito úteis, como bordas, texturas, contraste, formas alongadas e objetos finos, que combinam bem com rachaduras.

Na prática, reaproveitar um backbone já treinado era a opção com maior chance de trazer bons resultados com menor complexidade e menor risco técnico.

## Estratégia escolhida

A abordagem adotada foi utilizar o YOLOv8 como ponto de partida e executar fine-tuning em cima dos pesos existentes.

Isso significa que o modelo não começou aprendendo tudo do zero. Em vez disso, ele aproveitou o conhecimento visual prévio e foi ajustado para reconhecer a classe de interesse no novo domínio.

Essa estratégia é especialmente adequada quando:

- o dataset tem tamanho moderado ou limitado;
- o problema tem uma estrutura visual semelhante a padrões que modelos gerais já aprendem;
- existe a necessidade de chegar a uma solução funcional com boa relação entre esforço e resultado.

## Tipo de fine-tuning utilizado

O fine-tuning foi feito de forma supervisionada, usando os dados rotulados fornecidos. Em termos práticos, isso significa que o modelo foi atualizado com exemplos anotados de rachaduras para aprender a localizar e contar essas regiões nas imagens.

O fluxo geral seguiu esta lógica:

1. Carregamento de um modelo YOLO já treinado previamente.
2. Uso das imagens anotadas como base de adaptação.
3. Atualização dos pesos para especialização no novo conjunto de dados.
4. Geração do arquivo final `best.pt`, usado na inferência local e no microserviço.

Como o resultado final é usado para inferência com máscara e contagem, a modelagem é compatível com a tarefa de segmentação/detecção aplicada ao problema das rachaduras.

## Por que não treinar do zero

Treinar um modelo do zero poderia até ser uma opção teórica, mas neste caso era menos atrativa por vários motivos:

- exigiria um volume maior de imagens para generalizar bem;
- aumentaria o custo de treino;
- tenderia a demorar mais para convergir;
- teria maior chance de aprender padrões ruins caso o dataset não fosse grande o bastante;
- tornaria o projeto mais complexo sem garantia de ganho real sobre o fine-tuning.

Em resumo, com os dados disponíveis, o fine-tuning em YOLOv8 era a alternativa mais segura e pragmática.

## Resultados observados

Os resultados observados no notebook e na inferência local mostraram que o modelo consegue:

- localizar rachaduras em imagens de entrada;
- produzir a imagem anotada com a máscara/destaque da detecção;
- contar as ocorrências detectadas;
- separar visualmente o resultado da original de forma clara.

Na prática, isso confirmou que o fine-tuning conseguiu adaptar o YOLO ao domínio do problema sem exigir um pipeline de treino mais pesado ou mais arriscado.

Sem depender de métricas numéricas específicas aqui, o ponto principal é que o modelo final ficou útil para inferência local e também suficientemente estável para ser empacotado em um serviço web.

## POC em FastAPI

Depois de validar o pipeline de inferência, foi criada uma POC em FastAPI para transformar o fluxo de notebook em um microserviço consumível por HTTP.

Essa POC segue boas práticas básicas:

- carregamento do modelo uma vez no startup da aplicação;
- uso de endpoint assíncrono para receber a imagem;
- execução da inferência fora do event loop para não bloquear o servidor;
- retorno estruturado com contagem, detalhes das detecções e imagem processada.

### Endpoint principal

O serviço expõe o endpoint `POST /api/predict`, que recebe uma imagem e devolve:

- nome do arquivo;
- quantidade de rachaduras detectadas;
- lista de detecções com confiança e bounding box;
- imagem anotada em formato de data URL.

### Interface web simples

Além da API, foi criada uma interface web básica para facilitar a demonstração.

A interface permite:

- enviar uma imagem pelo navegador;
- visualizar o resultado processado;
- ver a contagem das rachaduras;
- inspecionar rapidamente as detecções retornadas pelo serviço.

O frontend é simples de propósito: o foco da entrega era demonstrar o fluxo completo de inferência, não construir uma aplicação visual complexa.

## Empacotamento e execução

O projeto também foi preparado para execução local e em container Docker.

Isso ajuda a garantir que o fluxo seja reproduzível e facilita a demonstração em qualquer ambiente compatível.

## Conclusão

A decisão de usar fine-tuning em YOLOv8 foi a mais adequada para o contexto do projeto. Havia imagens rotuladas em quantidade suficiente para especializar um modelo existente, mas não em volume ideal para justificar um treino do zero.

Ao aproveitar os padrões já aprendidos pelo YOLO, foi possível chegar a uma solução mais segura, mais simples de manter e com boa qualidade de resultado para o problema de detecção de rachaduras.

A POC em FastAPI com frontend simples fechou o ciclo do projeto ao transformar a inferência em um serviço acessível por navegador e por endpoint HTTP.
