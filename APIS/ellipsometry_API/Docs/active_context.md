# Contexto Activo del Proyecto TMM-R

Este archivo registra el estado actual del desarrollo del proyecto de simulación elipsométrica, sus cambios recientes y la arquitectura vigente.

## 1. Cambios Recientes (Última Sesión)

- **Reversión del Mapeo de Doble Fase (Autorange):** Se eliminó la optimización gruesa-fina (`autorange_fit_ellipsometry_torch`) de `tmm_utils_Rodrigo.py` debido a que no ofrecía ventajas de precisión y añadía complejidad. El sistema volvió a usar únicamente `fit_ellipsometry_torch`.
- **Estructuración de Documentación:** Se creó el archivo `agents.md` que sirve como mapa y guía de referencia de la arquitectura por capas para los agentes de IA.
- **Creación de Skills de Automatización:** Se implementó un plugin de skills del sistema (`C:\Users\mentu\.gemini\config\plugins\tmm\`) con herramientas personalizadas:
  - `tmm-debug-nan`: Checklist de depuración en caso de pérdida NaN.
  - `tmm-add-material`: Automatización para ingesta de nuevos materiales ópticos.
  - `tmm-analyze-fit`: Rutinas de graficado de resultados de ajuste y dispersión $n(\lambda)$.
  - `tmm-export-results`: Generación de reportes listos para incluir en la tesis (LaTeX, CSV, Markdown).
  - `git-selective-commit`: Automatización de commits con exclusión selectiva de archivos.
  - `Updater`: Mantenedor automático de este archivo.

## 2. Estado Actual de Componentes

| Capa | Estado y Detalles |
|---|---|
| **Física (`tmm_core.py`)** | Estable. Contiene algoritmos vectorizados e implementaciones diferenciables con PyTorch (`complex128`). |
| **Numérica (`tmm_utils_Rodrigo.py`)** | Limpia. Sin la función `autorange`. Mapea logits a espesores y parámetros físicos usando la función sigmoide. |
| **Ajuste (`fit_elipsometrico.py`)** | Orquesta la simulación. Consume `fit_ellipsometry_torch` directamente. |
| **API (`main.py`)** | FastAPI. Se removieron los parámetros de `autorange` en los esquemas Pydantic (`SimParam`). Conecta a MySQL. |
| **Persistencia (`models.py`)** | Sin cambios. Almacena usuarios y stacks optimizados. |

## 3. Próximos Pasos y Tareas Pendientes

- Derivar e implementar la formulación matemática para la **propagación del error** en `fit_ellipsometry_torch`.
- Validar el comportamiento de `f_air` en el modelo Bruggeman y habilitar opcionalmente su optimización si el usuario proporciona límites en la configuración.
