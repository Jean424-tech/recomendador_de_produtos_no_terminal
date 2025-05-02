# recomendador_de_produtos_no_terminal
Sistema de recomendação de produtos baseado em similaridade textual (TF-IDF e Levenshtein), tamanho físico e atributos como grupo e marca. Ideal para encontrar itens similares em catálogos ou sistemas de vendas a partir de um SKU consultado.

Este projeto é um script Python que realiza **recomendações de produtos similares** com base em **similaridade textual** (TF-IDF + Levenshtein), além de atributos como **grupo**, **marca** e **tamanho físico** extraído do nome. É ideal para auxiliar na identificação de produtos substitutos ou similares dentro de uma base de dados.

---

## 🔧 Tecnologias Utilizadas

- Python 3.8+
- Pandas
- Scikit-learn
- python-Levenshtein
- Regex
- UnicodeData

---

## 📂 Estrutura esperada do CSV (`seu_cadastro.csv`)

O script espera um arquivo `.csv` com `;` como separador e as seguintes colunas:

- `SKU` — Identificador único do produto
- `Nome` — Nome completo do produto
- `Marca` — Fabricante ou marca
- `Grupo` — Categoria do produto
- `GrupoPai` — Grupo pai (usado para lógica condicional)
- `Venda` — Volume ou índice de vendas (usado para ordenação)

### Exemplo de conteúdo:

```csv
SKU;Nome;Marca;Grupo;GrupoPai;Venda
1001;Cabo HDMI 2m;Sony;101;550000;120
1002;Cabo HDMI 1.5m;Sony;101;550000;100
1003;Adaptador USB-C para HDMI;Apple;102;400000;80
1004;Cabo HDMI 3m;Samsung;101;550000;90

