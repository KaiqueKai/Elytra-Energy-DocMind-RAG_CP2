# Relatório RAGAS — Elytra Energy (CKP02)

Perguntas por configuração: 5. Valores gerados pelo RAGAS nesta execução.

## Médias por configuração de chunking

| Chunk size | Overlap | Faithfulness média | Answer relevancy média | Perguntas com faithfulness |
|---:|---:|---:|---:|---:|
| 512 | 61 | 0.975 | 0.495 | 5/5 |
| 1024 | 123 | 1.000 | 0.716 | 5/5 |

## Resultado por pergunta

| Chunk | # | Faithfulness | Answer relevancy | Pergunta |
|---:|---:|---:|---:|---|
| 512 | 1 | 1.000 | 0.654 | Quais sistemas e equipamentos são tratados no Guia para Eficiência Energética nas Edificações Públicas do MME? |
| 512 | 2 | 1.000 | 0.854 | O que a etiqueta de eficiência energética do Inmetro (PBE) informa sobre os condicionadores de ar? |
| 512 | 3 | 1.000 | 0.000 | Que informações constam na tabela do Inmetro para refrigeradores e como usá-las ao comparar modelos? |
| 512 | 4 | 1.000 | 0.967 | Como a EPE acompanha o progresso da eficiência energética no Brasil no Atlas da Eficiência Energética? |
| 512 | 5 | 0.875 | 0.000 | O que o Inmetro informa sobre ventiladores de mesa, coluna e circuladores de ar na tabela de eficiência energética? |
| 1024 | 1 | 1.000 | 0.722 | Quais sistemas e equipamentos são tratados no Guia para Eficiência Energética nas Edificações Públicas do MME? |
| 1024 | 2 | 1.000 | 0.000 | O que a etiqueta de eficiência energética do Inmetro (PBE) informa sobre os condicionadores de ar? |
| 1024 | 3 | 1.000 | 0.963 | Que informações constam na tabela do Inmetro para refrigeradores e como usá-las ao comparar modelos? |
| 1024 | 4 | 1.000 | 0.950 | Como a EPE acompanha o progresso da eficiência energética no Brasil no Atlas da Eficiência Energética? |
| 1024 | 5 | 1.000 | 0.945 | O que o Inmetro informa sobre ventiladores de mesa, coluna e circuladores de ar na tabela de eficiência energética? |

## Configuração escolhida

Pelos dados desta execução, a melhor configuração é **chunk_size=1024** (faithfulness média 1.000; answer relevancy média 0.716). O critério foi maior faithfulness média, com answer relevancy como desempate.

Faithfulness média na zona ideal (≥ 0,9).
