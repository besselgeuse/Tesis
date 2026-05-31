pushd ..\..\venv\Scripts
call activate
popd

rem Ensure current directory is in PYTHONPATH for module imports
set PYTHONPATH=%CD%

echo Iniciando el navegador y el servidor...
start "" "http://127.0.0.1:8000"
uvicorn main:app --reload
