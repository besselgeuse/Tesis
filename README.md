# Multilayer Ellipsometry & TMM Simulation API ☀️

Anatomía de un motor óptico asincrónico para la simulación de películas delgadas antirreflectantes en celdas solares mediante el formalismo de matrices de transferencia.

## 🚀 Características Clave
- **Cálculo Diferenciable:** Implementación del formalismo analítico de Fresnel y matrices de transferencia empleando tensores complejos en PyTorch (`complex128`) con optimización por gradiente (Autograd).
- **Persistencia Escalable:** Almacenamiento transaccional de usuarios, límites de espesor (*bounds*) e historial de simulaciones optimizado mediante índices en MySQL.
- **Arquitectura Segura:** Flujo de autenticación completo utilizando el estándar OAuth2 con firma criptográfica de tokens JWT.

## 🛠️ Instalación y Despliegue

El proyecto está completamente contenedorizado de modo que no requiere configuraciones locales de bases de datos.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/Tesis.git](https://github.com/tu-usuario/Tesis.git)
   cd ellipsometry_API
