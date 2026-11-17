"""Assistente local de protocolos de incêndio (Defesa Civil).

Instalado via pip, este diretório vira o pacote `defesacivil_llm`; nos notebooks continua `src`.
"""
from .llm import DefesaCivilLLM, mascarar_pii

__version__ = "0.1.0"
__all__ = ["DefesaCivilLLM", "mascarar_pii", "__version__"]
