import pandas as pd
import unicodedata
import re
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Normaliza o texto removendo acentos e transformando para minúsculas
def normalizar_texto(texto):
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)).lower()
    texto = re.sub(r'\b(de|para|com|e|a|o|os|as)\b', '', texto)
    return texto.strip()

# Extrai tamanhos como "5m", "10cm", "12" etc.
def extrair_tamanho(nome):
    match = re.findall(r'(\d{1,3})(\s?\"|\spol|\sp|\smm|\scm|\sm)', nome.lower())
    if match:
        return int(match[0][0])
    return None

# Carrega o dataset
try:
    produtos_df = pd.read_csv('seu_cadastro.csv', encoding='utf-8', on_bad_lines='skip', dtype=str, sep=';')
except FileNotFoundError:
    print("Arquivo 'seu_cadastro.csv' não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    exit()

# Validação das colunas
colunas_necessarias = {'SKU', 'Nome', 'Marca', 'Grupo', 'GrupoPai', 'Venda'}
if not colunas_necessarias.issubset(produtos_df.columns):
    print(f"Erro: O arquivo CSV deve conter as colunas {colunas_necessarias}. Encontradas: {set(produtos_df.columns)}")
    exit()

# Normaliza os nomes e extrai tamanho
produtos_df = produtos_df.dropna(subset=['Nome'])
produtos_df['Nome_Normalizado'] = produtos_df['Nome'].apply(normalizar_texto)
produtos_df['Tamanho'] = produtos_df['Nome'].apply(extrair_tamanho)

def calcular_similaridade(produto_sku, produtos_df):
    if produto_sku not in produtos_df['SKU'].values:
        print(f"SKU '{produto_sku}' não encontrado.")
        return [], "Ruim"

    produto = produtos_df[produtos_df['SKU'] == produto_sku].iloc[0]
    nome_ref = produto['Nome_Normalizado']

    # Cria TF-IDF apenas entre o produto e os demais
    outros_produtos = produtos_df[produtos_df['SKU'] != produto_sku]
    corpus = [nome_ref] + outros_produtos['Nome_Normalizado'].tolist()

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)

    ref_vector = tfidf_matrix[0]
    outros_vectors = tfidf_matrix[1:]

    cosine_sim = cosine_similarity(ref_vector, outros_vectors)[0]

    sim_scores_cosseno = dict(zip(outros_produtos['SKU'], cosine_sim))

    sim_scores_levenshtein = {}
    for _, row in outros_produtos.iterrows():
        dist = levenshtein_distance(nome_ref, row['Nome_Normalizado'])
        sim_scores_levenshtein[row['SKU']] = 1 - (dist / max(len(nome_ref), len(row['Nome_Normalizado'])))

    scaler = MinMaxScaler()
    cos_values = list(sim_scores_cosseno.values())
    lev_values = list(sim_scores_levenshtein.values())
    if cos_values and lev_values:
        scaled_cos = scaler.fit_transform([[v] for v in cos_values])
        scaled_lev = scaler.fit_transform([[v] for v in lev_values])
        sim_scores_combinados = {
            sku: 0.6 * scaled_cos[i][0] + 0.4 * scaled_lev[i][0]
            for i, sku in enumerate(sim_scores_cosseno.keys())
        }
    else:
        sim_scores_combinados = {}

    return sorted(sim_scores_combinados.items(), key=lambda x: x[1], reverse=True), "Bom"

def recomendar_produtos(produto_sku, produtos_df, max_recomendacoes=5):
    produto = produtos_df[produtos_df['SKU'] == produto_sku].iloc[0]
    grupo_pai = produto['GrupoPai']

    if grupo_pai == '550000':
        tamanho_ref = extrair_tamanho(produto['Nome'])
        grupo_ref = produto['Grupo']

        similares = produtos_df[
            (produtos_df['Grupo'] == grupo_ref) &
            (produtos_df['SKU'] != produto_sku) &
            (produtos_df['Tamanho'].notna())
        ].copy()

        similares['Dif_Tamanho'] = similares['Tamanho'].astype(int).apply(lambda x: abs(x - tamanho_ref))
        similares = similares.sort_values(by=['Dif_Tamanho', 'Tamanho', 'Venda'], ascending=[True, True, True])

        return [(row['SKU'], row['Nome'], 1.0) for _, row in similares.head(max_recomendacoes).iterrows()]
    else:
        recomendacoes, _ = calcular_similaridade(produto_sku, produtos_df)
        produto = produtos_df[produtos_df['SKU'] == produto_sku].iloc[0]
        grupo_ref = produto['Grupo']
        marca_ref = produto['Marca']

        recomendacoes_filtradas = [
            (sku, produtos_df[produtos_df['SKU'] == sku]['Nome'].values[0], score)
            for sku, score in recomendacoes
            if sku != produto_sku and produtos_df[produtos_df['SKU'] == sku]['Grupo'].values[0] == grupo_ref
        ]

        mesma_marca = [
            rec for rec in recomendacoes_filtradas
            if produtos_df[produtos_df['SKU'] == rec[0]]['Marca'].values[0] == marca_ref
        ]

        outras_marcas = [
            rec for rec in recomendacoes_filtradas
            if produtos_df[produtos_df['SKU'] == rec[0]]['Marca'].values[0] != marca_ref
        ]

        resultado_final = []
        if mesma_marca:
            resultado_final.append(mesma_marca[0])
        resultado_final.extend(outras_marcas[:4])
        if len(resultado_final) < max_recomendacoes:
            resultado_final.extend(mesma_marca[1:max_recomendacoes - len(resultado_final)])
        if len(resultado_final) < max_recomendacoes:
            resultado_final.extend(outras_marcas[:max_recomendacoes - len(resultado_final)])

        return resultado_final[:max_recomendacoes]

# Loop principal
while True:
    print("\n=== RECOMENDADOR DE PRODUTOS ===")
    produto_sku = input("Digite o SKU do produto (ou 'sair' para encerrar): ")
    if produto_sku.lower() == 'sair':
        print("Encerrando...")
        break

    if produto_sku not in produtos_df['SKU'].values:
        print("SKU não encontrado.")
        continue

    produto_pesquisado = produtos_df[produtos_df['SKU'] == produto_sku].iloc[0]
    venda_exibicao = produto_pesquisado['Venda'] if 'Venda' in produto_pesquisado else 'N/A'

    print("\nProduto Consultado:")
    print(f"SKU: {produto_pesquisado['SKU']} | Nome: {produto_pesquisado['Nome']} | Venda: \033[92m{venda_exibicao}\033[0m")
    print("="*120)
    print("Recomendações aprimoradas:")

    recomendacoes = recomendar_produtos(produto_sku, produtos_df)
    for sku, nome, score in recomendacoes:
        venda = produtos_df[produtos_df['SKU'] == sku]['Venda'].values[0] if 'Venda' in produtos_df.columns else 'N/A'
        print(f"SKU: {sku} | Nome: {nome} | Venda: \033[92m{venda}\033[0m")

    print("="*120)
