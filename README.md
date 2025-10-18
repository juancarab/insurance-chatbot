# insurance-chatbot

Repo del proyecto final del curso de IA.  
Flujo: cada compañero crea rama/PR contra `develop`. Vos revisás y aprobás.

## Cómo empezar
1. Hacé fork o creá rama `feature/<tu-tema>` desde `develop`.
2. Instalá dependencias y pre-commit (ver abajo).
3. Abrí PR a `develop` usando el template, esperando CI verde y revisión.

## Comandos (local, opcional)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
pytest
