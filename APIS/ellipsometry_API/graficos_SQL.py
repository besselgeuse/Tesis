import models
from database import engine, SessionLocal
import numpy as np
import matplotlib.pyplot as plt
from tmm_utils_Rodrigo import cauchy_fn, constant_fn, load_interp, calculate_RT

STACK_ID = 22

def load_stack(stack_id: int):
    db = SessionLocal()
    try:
        stack = db.query(models.Stack).filter(models.Stack.id == stack_id).first()
        if stack is None:
            raise ValueError(f"Stack con id={stack_id} no encontrado.")
        return stack
    finally:
        db.close()

def plot_ellipsometry(stack):
    wl = np.array(stack.wl_exp)
    best_Ic = np.array(stack.best_Ic)
    best_Is = np.array(stack.best_Is)
    Ic_exp = np.array(stack.Ic_exp)
    Is_exp = np.array(stack.Is_exp)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(wl, Ic_exp, 'k-', label="Ic Experimental")
    ax[0].plot(wl, best_Ic, 'r--', label="Ic Calculada")
    ax[0].set_xlabel("Longitud de onda (nm)")
    ax[0].set_ylabel("Ic")
    ax[0].legend()
    ax[0].grid(True, linestyle=':', alpha=0.6)

    ax[1].plot(wl, Is_exp, 'k-', label="Is Experimental")
    ax[1].plot(wl, best_Is, 'r--', label="Is Calculada")
    ax[1].set_xlabel("Longitud de onda (nm)")
    ax[1].set_ylabel("Is")
    ax[1].legend()
    ax[1].grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(f"Stack: {stack.name} (id={stack.id})", fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_refractive_indices(stack):
    wl = np.array(stack.wl_exp)

    for layer in stack.layers:
        name = layer.get("name", "unknown")
        model = layer.get("model", "")
        thickness = layer.get("thickness", None)
        if model != "cauchy":
            continue
        A = layer.get("A")
        B = layer.get("B")
        C = layer.get("C")
        if A is None or B is None or C is None:
            continue
        n_fn = cauchy_fn(A, B, C)
        n = n_fn(wl)
        label = f"n {name}" + (f" ({thickness} nm)" if thickness else "")
        plt.figure(figsize=(10, 6))
        plt.plot(wl, n.real, 'k-', label=label)
        plt.xlim([np.min(wl), np.max(wl)])
        plt.ylim([np.min(n.real) - 0.1, np.max(n.real) + 0.1])
        plt.xlabel("Longitud de onda (nm)")
        plt.ylabel("Índice de refracción n")
        plt.title(f"Índice de refracción - {name} (Stack: {stack.name})", fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

def plot_compare_refractive_indices(stack):
    wl = np.array(stack.wl_exp)
    plt.figure(figsize=(10, 8))

    for layer in stack.layers:
        name = layer.get("name", "unknown")
        model = layer.get("model", "")
        if model != "cauchy":
            continue
        A = layer.get("A")
        B = layer.get("B")
        C = layer.get("C")
        if A is None or B is None or C is None:
            continue
        n_fn = cauchy_fn(A, B, C)
        n = n_fn(wl)
        plt.plot(wl, n.real, label=f"n {name}")

    plt.xlim([np.min(wl), np.max(wl)])
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Índice de refracción n")
    plt.title(f"Comparación de índices - Stack: {stack.name}", fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_reflectance_tmm(stack):
    wl = np.array(stack.wl_exp)

    materials = {}
    for layer in stack.layers:
        name = layer.get("name")
        model = layer.get("model", "")
        if name == "air":
            materials["air"] = (constant_fn(1.0), constant_fn(0.0))
        elif name == "Si":
            n_fn, k_fn = load_interp('./indices/nkdata/optical/Si.nk', skiprows=1)
            materials["Si"] = (n_fn, k_fn)
        elif model == "cauchy":
            A = layer.get("A")
            B = layer.get("B")
            C = layer.get("C")
            if A is not None and B is not None and C is not None:
                materials[name] = (cauchy_fn(A, B, C), constant_fn(0.0))

    stack_def = []
    for layer in stack.layers:
        name = layer.get("name")
        thickness = layer.get("thickness", np.inf)
        coherence = "i" if (thickness is None or thickness == np.inf or thickness == float("inf")) else "c"
        stack_def.append([thickness if thickness else np.inf, name, coherence])

    RT = calculate_RT(th_0=0, stack=stack_def, materials=materials, lams=wl)
    R_calc, _ = RT

    try:
        R_exp_path = f"./Datos/mediciones_de_reflectancia/{stack.name}.txt"
        R_exp = np.loadtxt(R_exp_path)
    except Exception:
        R_exp = None

    plt.figure(figsize=(10, 6))
    if R_exp is not None:
        plt.plot(wl, R_exp, 'k-', label="Reflectancia Experimental", linewidth=1.5)
    plt.plot(wl, R_calc, 'r--', label="Reflectancia Calculada (TMM)", linewidth=2)
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Reflectancia")
    plt.title(f"Reflectancia - Stack: {stack.name}", fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim([450, 850])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    stack = load_stack(STACK_ID)
    print(f"Stack cargado: {stack.name} (id={stack.id})")

    plot_ellipsometry(stack)
    plot_refractive_indices(stack)
    plot_compare_refractive_indices(stack)
    plot_reflectance_tmm(stack)
