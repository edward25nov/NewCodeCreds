"""Funciones compartidas por los notebooks de la Actividad 05.

Proyecto NewCodeCreds: desempeño de detectores de credenciales en repositorios
y tipos de archivo no vistos (CredData, commit 86a9182).
"""
import binascii
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

SEMILLA = 42
REPO_DOMINANTE = "2ba83c6a"  # OpenSSL, 33% de los casos T

# Extensión -> lenguaje, solo para los ocho lenguajes con casos suficientes
LENGUAJES = {
    ".go": "Go",
    ".py": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".java": "Java",
    ".rb": "Ruby",
    ".cs": "C#",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".php": "PHP",
}
CONFIG_TEXTO = {".json", ".yml", ".yaml", ".md", ".txt", ".xml", ".ini",
                ".properties", ".toml", ".cfg", ".conf", ".env"}


def id_corto(repo_id_largo: str) -> str:
    """Mismo mapeo que download_data.py: CRC32 del id del snapshot."""
    return f"{binascii.crc32(binascii.unhexlify(repo_id_largo)):08x}"


def mapa_organizaciones(snapshot_path: Path) -> pd.DataFrame:
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    filas = []
    for largo, url in snap.items():
        partes = url.rstrip("/").split("/")
        filas.append({"RepoName": id_corto(largo), "url": url,
                      "organizacion": partes[-2].lower()})
    return pd.DataFrame(filas)


def lineas_de_archivo(texto: str) -> list:
    """Divide un archivo en líneas con la misma regla que CredData (review_data.py).

    CredData numera las líneas tras convertir \\r\\n y \\r en \\n y cortar solo por \\n.
    No se usa str.splitlines(), que también corta en \\x0b, \\x0c, \\x1c-\\x1e, \\x85,
    \\u2028 y \\u2029 y desalinea el número de línea respecto a LineStart.
    """
    return texto.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def grupo_tipo_archivo(ext: str) -> str:
    if ext in LENGUAJES:
        return LENGUAJES[ext]
    if ext in CONFIG_TEXTO or ext == "":
        return "config_texto"
    return "otros"


# ---------------------------------------------------------------------
# Características de la línea para Random Forest
# ---------------------------------------------------------------------
PALABRAS = ["password", "passwd", "pwd", "secret", "token", "key", "api",
            "auth", "private", "credential", "bearer", "-----begin", "aws",
            "client", "access", "example", "test", "dummy", "fake", "xxx",
            "todo", "placeholder", "changeme", "sample"]


def entropia(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def caracteristicas_manuales(lineas: pd.Series) -> pd.DataFrame:
    """Rasgos numéricos de cada línea candidata (sin usar etiquetas ni metadatos)."""
    filas = []
    for s in lineas.fillna(""):
        tokens = [t for t in
                  s.replace('"', " ").replace("'", " ").replace("=", " ")
                   .replace(":", " ").replace(",", " ").split() if t]
        largo_tok = max((len(t) for t in tokens), default=0)
        tok_max = max(tokens, key=len) if tokens else ""
        n = max(len(s), 1)
        baja = s.lower()
        fila = {
            "largo": len(s),
            "n_tokens": len(tokens),
            "largo_token_max": largo_tok,
            "entropia_linea": entropia(s),
            "entropia_token_max": entropia(tok_max),
            "prop_digitos": sum(ch.isdigit() for ch in s) / n,
            "prop_mayus": sum(ch.isupper() for ch in s) / n,
            "prop_simbolos": sum((not ch.isalnum()) and (not ch.isspace()) for ch in s) / n,
            "tiene_asignacion": int(("=" in s) or (":" in s)),
            "tiene_comillas": int(('"' in s) or ("'" in s)),
        }
        for p in PALABRAS:
            fila[f"kw_{p.strip('-')}"] = int(p in baja)
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------
def metricas(y, pred, score=None) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    out = {
        "n": len(y),
        "positivos": int(y.sum()),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(set(y)) > 1 else np.nan,
    }
    out["fuga_residual"] = 1 - out["recall"]
    if score is not None and len(set(y)) > 1:
        out["pr_auc"] = average_precision_score(y, score)
    else:
        out["pr_auc"] = np.nan
    return out


def bootstrap_pareado(df, col_y, pred_a, pred_b, grupos, metrica="f1",
                      n_boot=1000, semilla=SEMILLA):
    """Diferencia (b - a) de una métrica sobre las MISMAS líneas.

    Remuestrea por grupo (organización), porque las líneas de un mismo
    proyecto no son independientes. Devuelve la diferencia observada y el IC 95%.
    """
    fn = {"f1": lambda y, p: f1_score(y, p, zero_division=0),
          "mcc": matthews_corrcoef,
          "recall": lambda y, p: recall_score(y, p, zero_division=0),
          "precision": lambda y, p: precision_score(y, p, zero_division=0)}[metrica]
    y = df[col_y].to_numpy()
    a = df[pred_a].to_numpy()
    b = df[pred_b].to_numpy()
    g = df[grupos].to_numpy()
    obs = fn(y, b) - fn(y, a)
    unicos = np.unique(g)
    idx_por_grupo = {u: np.flatnonzero(g == u) for u in unicos}
    rng = np.random.default_rng(semilla)
    difs = []
    for _ in range(n_boot):
        elegidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([idx_por_grupo[u] for u in elegidos])
        if len(set(y[idx])) < 2:
            continue
        difs.append(fn(y[idx], b[idx]) - fn(y[idx], a[idx]))
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return {"metrica": metrica, "diferencia": obs, "ic95_inf": lo,
            "ic95_sup": hi, "n_boot": len(difs)}
