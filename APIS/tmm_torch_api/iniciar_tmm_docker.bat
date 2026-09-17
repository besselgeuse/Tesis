@echo off
cd /d "%~dp0"
echo Iniciando base de datos y API en Docker...
:: Iniciar en segundo plano (-d)
docker compose up -d

echo Esperando que el servidor web este listo...
timeout /t 5 /nobreak > nul

echo Iniciando tu navegador en http://localhost:8000...
start "" "http://localhost:8000"

echo.
echo Para apagar el sistema, ejecuta 'apagar_tmm_docker.bat' o usa Docker Desktop.
pause
