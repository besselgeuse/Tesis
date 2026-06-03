@echo off
cd /d "%~dp0"
echo Apagando contenedores de la API de TMM...
docker compose down
echo Todo se ha detenido correctamente.
pause
