## Multilayer Ellipsometry & TMM Simulation API ☀️

Este codigo tiene el objetivo de fitear parametros elipsometricos de peliculas delgadas de materiales antirreflectantes usados en celdas solares. Para ello se usan principalmente dos librerias, tmm y pytorch. La primera se encarga del calculo de los coeficientes de fresnel de las peliculas usando el metodo de la matriz de transferencia, mientras que la segunda se encargar de encontrar el espesor y parametros de dispersión que mejor se ajustan a los datos experimentales usando su función autograd para disminuir la perdida, perdida que se definió usando el error cuadratico medio.

## 🚀 Características Clave
- **Cálculo Diferenciable:** Implementación del formalismo analítico de Fresnel y matrices de transferencia empleando tensores complejos en PyTorch (`complex128`) con optimización por gradiente (Autograd) y soporte para tecnologia CUDA.
- **Persistencia Escalable:** Almacenamiento transaccional de usuarios, límites de espesor (*bounds*) e historial de simulaciones optimizado mediante índices en MySQL.
- **Arquitectura Segura:** Flujo de autenticación completo utilizando el estándar OAuth2 con firma criptográfica de tokens JWT.   

## 🛠️ Instalación y Despliegue

El proyecto está completamente dockerizado, de modo que no requiere configuraciones locales de bases de datos.

1. **Clonar el repositorio:**
   - entrar a la terminal
   - git clone -b APIS https://github.com/besselgeuse/Tesis.git
   - cd APIS/ellipsometry_API
   - docker compose up --build
   - Entrar al http://localhost:8000 en el navegador
   - Listo!
 
## Modo de uso
<img width="1007" height="939" alt="fit_ellipsometrico" src="https://github.com/user-attachments/assets/aaefcc48-baa8-40d5-a5db-e18070665a45" />

- En la pestaña de stack optico se eligen las capas de materiales que conformaran la celda y sus correspondientes modelos de dispersión,en caso de no querer usar un modelo simplemente seleccionar estático.
- Es muy importante que el primer elemento siempre sea el aire y el ultimo un semiconductor como el Silicio, y que ambos sean estáticos.
- Si se elige el modelo de Bruggeman el parametro a ajuster es la fracción de aire que tiene el material.
- En la pestaña de parametros se puede ajustar el angulo de incidencia con el que fue medida la muestra, asi como los parametros de aprendizaje del autograd.
- Si bien tiene la opción de usar tecnologia CUDA en la versión dockerizada no está disponible puesto que se instaló una libreria de pytorch mas liviana que solo trabaja con la cpu.

## Archivos
En la pestaña de historial quedarán guardados los archivos con los parametros y datos ajustados para descargar en formato txt. Cada usuario tendra acceso unicamente a sus propias mediciones.
