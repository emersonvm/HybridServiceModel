import os
import time
import json
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# NOVO: detecção de idioma
from langdetect import detect_langs, DetectorFactory, LangDetectException
DetectorFactory.seed = 0  # resultados reprodutíveis

# -----------------------
# Configuração & segurança
# -----------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Defina OPENAI_API_KEY no arquivo .env")

client = OpenAI(api_key=API_KEY)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

INPUT_FILE = "test4k.csv" #------- aqui a variável INPUT_FILE é declarada recebendo o arquivo test4k.csv que possui 1920 linhas
OUTPUT_FILE = "saida_analise.csv"
TEMP_FILE = "saida_temp.csv"
COL_PERGUNTA = "subject"
COL_RESPOSTA = "answer"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def ler_csv_certo(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine='python',
            encoding='utf-8-sig',
            quotechar='"',
            on_bad_lines='skip',
            usecols=[COL_PERGUNTA, COL_RESPOSTA]
        )
        return df
    except Exception as e:
        logging.warning(f"Falha ao autodetectar separador: {e}. Tentando sep=';'.")
        df = pd.read_csv(
            path,
            sep=';',
            engine='python',
            encoding='utf-8-sig',
            quotechar='"',
            on_bad_lines='skip',
            usecols=[COL_PERGUNTA, COL_RESPOSTA]
        )
        return df

# ===== NOVO: utilitário de limpeza antes da detecção =====
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CODE_RE = re.compile(r"`[^`]+`|<[^>]+>")  # remove trechos marcados como código/HTML simples

def _preclean(text: str) -> str:
    t = str(text or "").strip()
    t = _URL_RE.sub(" ", t)
    t = _CODE_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t

# ===== NOVO: detecção e filtro de idioma =====
WHITELIST = {"en", "pt", "es"}
CONF_THRESHOLD = 0.90  # confiança mínima

def _safe_detect(text: str) -> str:
    """
    Detecta idioma com langdetect (com confiança).
    - Limpa URLs/trechos de código antes da detecção
    - Ignora textos muito curtos/vazios
    - Usa detect_langs e exige probabilidade >= CONF_THRESHOLD
    - Normaliza 'pt-br' para 'pt'
    """
    t = _preclean(text)
    if not t or len(t) < 12:
        return "unknown"
    try:
        langs = detect_langs(t)  # ex.: [en:0.99, de:0.01]
        if not langs:
            return "unknown"
        top = sorted(langs, key=lambda x: x.prob, reverse=True)[0]
        lang = top.lang.lower()
        prob = float(top.prob)
        if lang == "pt-br":
            lang = "pt"
        if prob >= CONF_THRESHOLD:
            return lang
        # Se a confiança do top-1 é baixa, marca como desconhecido
        return "unknown"
    except LangDetectException:
        return "unknown"

def filtrar_idiomas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lang"] = df[COL_PERGUNTA].fillna("").apply(_safe_detect)

    # Log da distribuição bruta
    dist = df["lang"].value_counts(dropna=False).to_dict()
    logging.info(f"Distribuição de idiomas detectados (antes do filtro): {dist}")

    # Aceita apenas português, inglês e espanhol
    filtrado = df[df["lang"].isin(WHITELIST)]

    removidos = len(df) - len(filtrado)
    logging.info(
        f"Filtro de idioma aplicado (permitidos: {sorted(WHITELIST)}). "
        f"{len(filtrado)} mantidos; {removidos} removidos."
    )
    return filtrado

def analisar_interacao(pergunta: str, resposta: str, max_retries: int = 5, base_sleep: float = 1.5):
    prompt = f"""
A seguir está uma interação entre um usuário e um atendente.

Pergunta: "{pergunta}"
Resposta: "{resposta}"

Avalie e responda em JSON com as chaves exatamente:
{{
  "sentimento_pergunta": "Positivo|Negativo|Neutro",
  "sentimento_resposta": "Positivo|Negativo|Neutro",
  "satisfacao": "Sim|Não",
  "motivo": "string curta explicando"
}}
Se algo estiver vazio, use "Neutro" e explique sucintamente em "motivo".
"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            msg = str(e)
            is_retryable = any(code in msg for code in ["429", "500", "502", "503", "504"])
            if is_retryable and attempt < max_retries - 1:
                sleep_s = base_sleep * (2 ** attempt)
                logging.warning(f"Erro temporário ({msg}). Retry em {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            logging.error(f"Falha definitiva na análise: {e}")
            return {
                "sentimento_pergunta": "Erro",
                "sentimento_resposta": "Erro",
                "satisfacao": "Erro",
                "motivo": msg[:500]
            }

def main():
    df = ler_csv_certo(INPUT_FILE)
    if df.empty:
        raise RuntimeError("CSV lido, mas sem linhas/colunas esperadas. Confere cabeçalho e separador.")

    # ===== aplicar filtro por idioma (en/pt/es) =====
    df = filtrar_idiomas(df)
    if df.empty:
        raise RuntimeError("Após filtrar por idioma, não restaram linhas (somente en/pt/es). Revise o CSV.")

    resultados = []
    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        pergunta = str(row.get(COL_PERGUNTA, "") or "")
        resposta = str(row.get(COL_RESPOSTA, "") or "")
        logging.info(f"Analisando linha {i}/{total} (idx original {idx})...")

        resultado = analisar_interacao(pergunta, resposta)
        resultados.append(resultado)

        # checkpoint periódico
        if (i % 10 == 0) or (i == total):
            df_parcial = pd.DataFrame(resultados)
            # Inclui as colunas originais + legíveis
            base = df.iloc[:i].reset_index(drop=True).copy()
            base["pergunta_original"] = base[COL_PERGUNTA]
            base["resposta_original"] = base[COL_RESPOSTA]
            df_temp = pd.concat([base, df_parcial], axis=1)
            # Reordena para garantir pergunta/resp no início
            cols_first = [
                COL_PERGUNTA, COL_RESPOSTA,
                "pergunta_original", "resposta_original",
                "lang"
            ]
            cols_rest = [c for c in df_temp.columns if c not in cols_first]
            df_temp = df_temp[cols_first + cols_rest]
            df_temp.to_csv(TEMP_FILE, index=False, encoding='utf-8-sig')
            logging.info(f"Checkpoint salvo em {TEMP_FILE} ({i} linhas).")

        time.sleep(0.8)

    # Final: concatena originais + análise
    df_analise = pd.DataFrame(resultados)
    base = df.reset_index(drop=True).copy()
    base["pergunta_original"] = base[COL_PERGUNTA]
    base["resposta_original"] = base[COL_RESPOSTA]
    df_final = pd.concat([base, df_analise], axis=1)

    # Reordena colunas para facilitar leitura
    cols_first = [
        COL_PERGUNTA, COL_RESPOSTA,
        "pergunta_original", "resposta_original",
        "lang",
        "sentimento_pergunta", "sentimento_resposta", "satisfacao", "motivo"
    ]
    cols_rest = [c for c in df_final.columns if c not in cols_first]
    df_final = df_final[cols_first + cols_rest]

    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    logging.info(f"Arquivo final salvo: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()