#%%
import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Stack_simulator as st
import os
import sys

# Configuración de apariencia
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Paleta de colores solicitada
COLOR_BTN_PRIMARY = "#FFB347" # Naranja claro
COLOR_BTN_HOVER_PRIMARY = "#FFA726"
COLOR_BTN_SECONDARY = "#9C27B0" # Morado
COLOR_BTN_HOVER_SECONDARY = "#7B1FA2"
TEXT_COLOR_DARK = "#000000"
TEXT_COLOR_LIGHT = "#FFFFFF"

class RedirectText(object):
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root

    def write(self, string):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", string)
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")
        self.root.update()

    def flush(self):
        pass

class SimuladorARC(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Simulador Óptico TMM - Optimización de ARC")
        self.geometry("1100x850")
        
        # Variables de estado
        self.sim_ref = None
        self.sim_lams = None
        self.sim_js_am0 = None
        self.sim_R_max = None
        self.sim_thicks = None

        # Frame principal scrolleable
        self.main_scroll = ctk.CTkScrollableFrame(self)
        self.main_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        self.build_config_section()
        self.build_sim_section()
        self.build_opt_section()
        self.build_save_section()
        self.build_comp_section()

    def build_config_section(self):
        self.frame_config = ctk.CTkFrame(self.main_scroll)
        self.frame_config.pack(fill="x", padx=10, pady=10)
        
        lbl_config_title = ctk.CTkLabel(self.frame_config, text="🔧 CONFIGURACIÓN DEL STACK Y SIMULACIÓN", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_BTN_SECONDARY)
        lbl_config_title.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.frame_config.grid_columnconfigure(0, weight=1)
        self.frame_config.grid_columnconfigure(1, weight=1)

        # Panel izquierdo (IQE, Caché y Stack)
        left_panel = ctk.CTkFrame(self.frame_config, fg_color="transparent")
        left_panel.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(left_panel, text="Ruta de datos IQE (.txt):", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        frame_iqe = ctk.CTkFrame(left_panel, fg_color="transparent")
        frame_iqe.pack(fill="x", pady=5)
        self.entry_iqe = ctk.CTkEntry(frame_iqe)
        self.entry_iqe.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_iqe.insert(0, "./indices/IQEGaAs2.txt")
        ctk.CTkButton(frame_iqe, text="Explorar", width=80, fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=self.browse_iqe).pack(side="left")

        # Archivo Caché
        ctk.CTkLabel(left_panel, text="Archivo Caché para Guardar/Cargar (.pickle) (opcional):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 0))
        frame_cache = ctk.CTkFrame(left_panel, fg_color="transparent")
        frame_cache.pack(fill="x", pady=5)
        self.entry_cache = ctk.CTkEntry(frame_cache, placeholder_text="Ej: ./Files/mi_cache.pickle")
        self.entry_cache.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(frame_cache, text="Explorar", width=80, fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=self.browse_cache).pack(side="left")

        # Checkbox No guardar
        self.var_no_guardar = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left_panel, text="No guardar", variable=self.var_no_guardar, text_color=TEXT_COLOR_LIGHT, fg_color=COLOR_BTN_PRIMARY, hover_color=COLOR_BTN_HOVER_PRIMARY).pack(anchor="w", pady=(5, 5))

        # Archivo Catálogo
        ctk.CTkLabel(left_panel, text="Ruta del Catálogo de Materiales (.txt):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 0))
        frame_catalog = ctk.CTkFrame(left_panel, fg_color="transparent")
        frame_catalog.pack(fill="x", pady=5)
        self.entry_catalog = ctk.CTkEntry(frame_catalog)
        self.entry_catalog.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_catalog.insert(0, "./catalogo_de_materiales.txt")
        ctk.CTkButton(frame_catalog, text="Explorar", width=80, fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=self.browse_catalog).pack(side="left")

        ctk.CTkLabel(left_panel, text="Configuración del Stack (Capas):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10,0))
        self.stack_scroll = ctk.CTkScrollableFrame(left_panel, height=150, label_text="Índice | Espesor | Material | Coherencia")
        self.stack_scroll.pack(fill="x", pady=5)
        self.stack_rows = []
        default_stack = [("inf", "air", "i"), ("50", "T1_porosa", "c"), ("50", "T1_densa", "c"), ("25", "InGaP", "c"), ("inf", "GaAs", "i")]
        for thick, mat, coh in default_stack:
            self.add_stack_row(thick, mat, coh)
        ctk.CTkButton(left_panel, text="+ Añadir Capa", fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=lambda: self.add_stack_row("", "", "c")).pack(fill="x")

        # Panel derecho (Thicks)
        right_panel = ctk.CTkFrame(self.frame_config, fg_color="transparent")
        right_panel.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(right_panel, text="Optimización de Espesores (Thicks):", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self.thicks_scroll = ctk.CTkScrollableFrame(right_panel, height=150, label_text="Índice Capa | Min | Max | Paso")
        self.thicks_scroll.pack(fill="x", pady=5)
        self.thick_rows = []
        self.add_thick_row("1", "10", "120", "1")
        self.add_thick_row("2", "10", "100", "1")
        ctk.CTkButton(right_panel, text="+ Añadir Rango", fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=lambda: self.add_thick_row("", "", "", "1")).pack(fill="x")

    def build_sim_section(self):
        self.frame_sim = ctk.CTkFrame(self.main_scroll)
        self.frame_sim.pack(fill="x", padx=10, pady=10)
        
        self.btn_run = ctk.CTkButton(self.frame_sim, text="▶ 1. Ejecutar Simulación", font=ctk.CTkFont(size=16, weight="bold"), 
                                     height=40, fg_color=COLOR_BTN_PRIMARY, text_color=TEXT_COLOR_DARK, hover_color=COLOR_BTN_HOVER_PRIMARY, command=self.run_simulation)
        self.btn_run.pack(fill="x", padx=20, pady=10)

        # Log TextBox para mostrar progreso
        self.log_textbox = ctk.CTkTextbox(self.frame_sim, height=120, state="disabled")
        self.log_textbox.pack(fill="x", padx=20, pady=5)
        
        self.canvas_sim_frame = ctk.CTkFrame(self.frame_sim, fg_color="transparent")
        self.canvas_sim_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def build_opt_section(self):
        self.frame_opt = ctk.CTkFrame(self.main_scroll)
        self.frame_opt.pack(fill="x", padx=10, pady=10)
        
        lbl_opt_title = ctk.CTkLabel(self.frame_opt, text="📈 2. GRAFICAR REFLECTANCIA ÓPTIMA", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_BTN_SECONDARY)
        lbl_opt_title.pack(pady=10)

        opt_controls = ctk.CTkFrame(self.frame_opt, fg_color="transparent")
        opt_controls.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(opt_controls, text="Cargar desde archivo (.txt) (opcional):", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        frame_opt_file = ctk.CTkFrame(opt_controls, fg_color="transparent")
        frame_opt_file.pack(fill="x", pady=5)
        self.entry_opt_file = ctk.CTkEntry(frame_opt_file, placeholder_text="Deja vacío para usar los de la simulación actual")
        self.entry_opt_file.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(frame_opt_file, text="Explorar", width=80, fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=self.browse_opt_file).pack(side="left")

        self.btn_opt = ctk.CTkButton(self.frame_opt, text="Graficar", font=ctk.CTkFont(size=16, weight="bold"), 
                                     height=40, fg_color=COLOR_BTN_PRIMARY, text_color=TEXT_COLOR_DARK, hover_color=COLOR_BTN_HOVER_PRIMARY, command=self.plot_optimal_reflectance)
        self.btn_opt.pack(fill="x", padx=20, pady=10)
        
        self.canvas_opt_frame = ctk.CTkFrame(self.frame_opt, fg_color="transparent")
        self.canvas_opt_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def build_save_section(self):
        self.frame_save = ctk.CTkFrame(self.main_scroll)
        self.frame_save.pack(fill="x", padx=10, pady=10)
        
        self.btn_save = ctk.CTkButton(self.frame_save, text="💾 3. Guardar Reflectancia", font=ctk.CTkFont(size=16, weight="bold"), 
                                      height=40, fg_color=COLOR_BTN_PRIMARY, text_color=TEXT_COLOR_DARK, hover_color=COLOR_BTN_HOVER_PRIMARY, command=self.save_reflectance)
        self.btn_save.pack(fill="x", padx=20, pady=10)

    def build_comp_section(self):
        self.frame_comp = ctk.CTkFrame(self.main_scroll)
        self.frame_comp.pack(fill="x", padx=10, pady=10)
        
        lbl_comp = ctk.CTkLabel(self.frame_comp, text="🔄 COMPARAR REFLECTANCIAS", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_BTN_SECONDARY)
        lbl_comp.pack(pady=10)
        
        comp_controls = ctk.CTkFrame(self.frame_comp, fg_color="transparent")
        comp_controls.pack(fill="x", padx=10, pady=5)
        
        # Archivo 1
        f1_frame = ctk.CTkFrame(comp_controls, fg_color="transparent")
        f1_frame.pack(fill="x", pady=2)
        self.entry_f1 = ctk.CTkEntry(f1_frame, placeholder_text="Archivo 1")
        self.entry_f1.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f1_frame, text="Buscar", width=60, fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=lambda: self.browse_comp(self.entry_f1)).pack(side="left")
        self.u1_var = ctk.StringVar(value="1")
        ctk.CTkOptionMenu(f1_frame, values=["1", "100"], variable=self.u1_var, width=60, fg_color=COLOR_BTN_SECONDARY, button_color=COLOR_BTN_HOVER_SECONDARY, button_hover_color=COLOR_BTN_SECONDARY).pack(side="left", padx=5)
        
        # Archivo 2
        f2_frame = ctk.CTkFrame(comp_controls, fg_color="transparent")
        f2_frame.pack(fill="x", pady=2)
        self.entry_f2 = ctk.CTkEntry(f2_frame, placeholder_text="Archivo 2")
        self.entry_f2.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(f2_frame, text="Buscar", width=60, fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, command=lambda: self.browse_comp(self.entry_f2)).pack(side="left")
        self.u2_var = ctk.StringVar(value="1")
        ctk.CTkOptionMenu(f2_frame, values=["1", "100"], variable=self.u2_var, width=60, fg_color=COLOR_BTN_SECONDARY, button_color=COLOR_BTN_HOVER_SECONDARY, button_hover_color=COLOR_BTN_SECONDARY).pack(side="left", padx=5)
        
        self.btn_comp = ctk.CTkButton(self.frame_comp, text="Comparar", font=ctk.CTkFont(weight="bold"), 
                                      fg_color=COLOR_BTN_PRIMARY, text_color=TEXT_COLOR_DARK, hover_color=COLOR_BTN_HOVER_PRIMARY, command=self.compare_reflectances)
        self.btn_comp.pack(pady=10)
        
        self.canvas_comp_frame = ctk.CTkFrame(self.frame_comp, fg_color="transparent")
        self.canvas_comp_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # --- MÉTODOS AUXILIARES UI ---

    def browse_iqe(self):
        filename = filedialog.askopenfilename(title="Seleccionar archivo IQE", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filename:
            self.entry_iqe.delete(0, ctk.END)
            self.entry_iqe.insert(0, filename)

    def browse_catalog(self):
        filename = filedialog.askopenfilename(title="Seleccionar catálogo de materiales", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filename:
            self.entry_catalog.delete(0, ctk.END)
            self.entry_catalog.insert(0, filename)
            self.update_material_comboboxes()
            
    def get_catalog_materials(self, filepath):
        import re
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            matches = re.findall(r"['\"](.*?)['\"]\s*:", content)
            return list(dict.fromkeys(matches))
        except:
            return []

    def update_material_comboboxes(self):
        mats = self.get_catalog_materials(self.entry_catalog.get().strip())
        if not mats:
            mats = ["air", "GaAs"]
        for row in self.stack_rows:
            row["mat"].configure(values=mats)

    def browse_cache(self):
        filename = filedialog.askopenfilename(title="Seleccionar archivo Caché", filetypes=[("Pickle Files", "*.pickle"), ("All Files", "*.*")])
        if not filename:
            # Si el usuario cancela o quiere guardar un archivo nuevo
            filename = filedialog.asksaveasfilename(title="Crear archivo Caché", defaultextension=".pickle", filetypes=[("Pickle Files", "*.pickle"), ("All Files", "*.*")])
        if filename:
            self.entry_cache.delete(0, ctk.END)
            self.entry_cache.insert(0, filename)

    def browse_opt_file(self):
        filename = filedialog.askopenfilename(title="Seleccionar archivo de reflectancia", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filename:
            self.entry_opt_file.delete(0, ctk.END)
            self.entry_opt_file.insert(0, filename)

    def browse_comp(self, entry_widget):
        filename = filedialog.askopenfilename(title="Seleccionar archivo", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filename:
            entry_widget.delete(0, ctk.END)
            entry_widget.insert(0, filename)

    def add_stack_row(self, d_val, mat_val, coh_val):
        idx = len(self.stack_rows)
        row_frame = ctk.CTkFrame(self.stack_scroll)
        row_frame.pack(fill="x", pady=2)
        
        lbl_idx = ctk.CTkLabel(row_frame, text=f"{idx}", width=30)
        lbl_idx.pack(side="left", padx=2)
        
        entry_d = ctk.CTkEntry(row_frame, width=50, placeholder_text="inf/nm")
        entry_d.insert(0, d_val)
        entry_d.pack(side="left", padx=2)
        
        mats = self.get_catalog_materials(self.entry_catalog.get().strip())
        if not mats:
            mats = ["air", "GaAs"]
            
        entry_mat = ctk.CTkComboBox(row_frame, width=100, values=mats)
        entry_mat.set(mat_val)
        entry_mat.pack(side="left", expand=True, fill="x", padx=2)
        
        combo_coh = ctk.CTkOptionMenu(row_frame, values=["i", "c"], width=50, fg_color=COLOR_BTN_SECONDARY, button_color=COLOR_BTN_HOVER_SECONDARY, button_hover_color=COLOR_BTN_SECONDARY)
        combo_coh.set(coh_val)
        combo_coh.pack(side="left", padx=2)
        btn_remove = ctk.CTkButton(row_frame, text="X", width=30, fg_color="#dc3545", hover_color="#c82333", command=lambda f=row_frame: self.remove_row(f, self.stack_rows))
        btn_remove.pack(side="right", padx=2)
        self.stack_rows.append({"frame": row_frame, "idx_lbl": lbl_idx, "d": entry_d, "mat": entry_mat, "coh": combo_coh})

    def add_thick_row(self, idx_val, min_val, max_val, step_val):
        if not idx_val:
            idx_val = str(len(self.thick_rows) + 1)
        row_frame = ctk.CTkFrame(self.thicks_scroll)
        row_frame.pack(fill="x", pady=2)
        entry_idx = ctk.CTkEntry(row_frame, width=40, placeholder_text="Idx")
        entry_idx.insert(0, idx_val)
        entry_idx.pack(side="left", padx=2)
        entry_min = ctk.CTkEntry(row_frame, width=40, placeholder_text="Min")
        entry_min.insert(0, min_val)
        entry_min.pack(side="left", expand=True, fill="x", padx=2)
        entry_max = ctk.CTkEntry(row_frame, width=40, placeholder_text="Max")
        entry_max.insert(0, max_val)
        entry_max.pack(side="left", expand=True, fill="x", padx=2)
        entry_step = ctk.CTkEntry(row_frame, width=40, placeholder_text="Paso")
        entry_step.insert(0, step_val)
        entry_step.pack(side="left", padx=2)
        btn_remove = ctk.CTkButton(row_frame, text="X", width=30, fg_color="#dc3545", hover_color="#c82333", command=lambda f=row_frame: self.remove_row(f, self.thick_rows))
        btn_remove.pack(side="right", padx=2)
        self.thick_rows.append({"frame": row_frame, "idx": entry_idx, "min": entry_min, "max": entry_max, "step": entry_step})

    def remove_row(self, frame, row_list):
        frame.destroy()
        row_list[:] = [row for row in row_list if row["frame"] != frame]
        # Re-indexar etiquetas si es la lista de capas (stack_rows)
        if row_list is self.stack_rows:
            for i, row in enumerate(self.stack_rows):
                row["idx_lbl"].configure(text=f"{i}")

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def embed_figures(self, figs, container, max_cols=2):
        self.clear_frame(container)
        for idx, fig in enumerate(figs):
            row = idx // max_cols
            col = idx % max_cols
            
            fig_frame = ctk.CTkFrame(container, fg_color="transparent")
            fig_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            canvas = FigureCanvasTkAgg(fig, master=fig_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill="both", expand=True)
            
            btn_save_img = ctk.CTkButton(fig_frame, text="Guardar Imagen", fg_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_HOVER_SECONDARY, 
                                         command=lambda f=fig: self.save_figure(f))
            btn_save_img.pack(pady=5)
            
            container.grid_columnconfigure(col, weight=1)

    def save_figure(self, fig):
        filepath = filedialog.asksaveasfilename(defaultextension=".png", 
                                                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf")])
        if filepath:
            try:
                fig.savefig(filepath, bbox_inches='tight')
                messagebox.showinfo("Éxito", "Imagen guardada correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen:\n{str(e)}")

    # --- MÉTODOS DE LÓGICA ---

    def run_simulation(self):
        try:
            ruta_iqe = self.entry_iqe.get()
            if not ruta_iqe: raise ValueError("La ruta IQE no puede estar vacía.")

            if self.var_no_guardar.get():
                file_dir = 'No_guardar'
            else:
                file_dir = self.entry_cache.get().strip()
                if not file_dir:
                    file_dir = 'None'

            stack = []
            for row in self.stack_rows:
                d_str = row["d"].get().strip().lower()
                mat = row["mat"].get().strip()
                coh = row["coh"].get().strip()
                d = np.inf if d_str in ['inf', 'np.inf'] else float(d_str)
                stack.append([d, mat, coh])

            thicks = {}
            for row in self.thick_rows:
                idx = int(row["idx"].get().strip())
                val_min = float(row["min"].get().strip())
                val_max = float(row["max"].get().strip())
                val_step = float(row["step"].get().strip())
                thicks[idx] = np.arange(val_min, val_max + val_step, val_step)

            self.btn_run.configure(text="Ejecutando...", state="disabled")
            
            # Limpiar Log
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.insert("end", "Iniciando simulación...\n")
            self.log_textbox.configure(state="disabled")
            self.update()

            # Redirigir stdout para capturar los prints de la terminal a la UI
            old_stdout = sys.stdout
            sys.stdout = RedirectText(self.log_textbox, self)

            try:
                res = st.ejecutar_simulacion(stack, ruta_iqe, thicks, File_direction=file_dir, ruta_catalogo=self.entry_catalog.get().strip())
            finally:
                # Restaurar stdout siempre
                sys.stdout = old_stdout

            self.sim_ref, trans, self.sim_lams, self.sim_js_am0, js_am15, self.sim_thicks, figs = res

            if figs:
                self.embed_figures(figs, self.canvas_sim_frame)

            messagebox.showinfo("Éxito", "Simulación completada.")
            
        except Exception as e:
            # Restaurar stdout en caso de error temprano
            if sys.stdout != sys.__stdout__:
                sys.stdout = sys.__stdout__
            messagebox.showerror("Error", f"Ha ocurrido un error:\n{str(e)}")
        finally:
            if sys.stdout != sys.__stdout__:
                sys.stdout = sys.__stdout__
            self.btn_run.configure(text="▶ 1. Ejecutar Simulación", state="normal")

    def plot_optimal_reflectance(self):
        try:
            archivo_cargar = self.entry_opt_file.get().strip()
            if archivo_cargar == "":
                archivo_cargar = None

            if archivo_cargar is None and (self.sim_ref is None or self.sim_lams is None or self.sim_js_am0 is None):
                messagebox.showwarning("Advertencia", "Debes ejecutar una simulación primero, o cargar un archivo .txt con reflectancia.")
                return

            if archivo_cargar is not None:
                self.sim_R_max, fig, self.sim_lams = st.graficar_reflectancia_optima(archivo=archivo_cargar)
                msg = "Datos cargados y graficados desde el archivo."
            else:
                self.sim_R_max, fig, self.sim_lams = st.graficar_reflectancia_optima(ref=self.sim_ref, lams=self.sim_lams, js_am0=self.sim_js_am0, thicks=self.sim_thicks)
                msg = "Reflectancia óptima calculada y graficada."

            self.embed_figures([fig], self.canvas_opt_frame, max_cols=1)
            
            if archivo_cargar is not None:
                messagebox.showinfo("Info", msg)

        except Exception as e:
            messagebox.showerror("Error", f"Ha ocurrido un error al graficar:\n{str(e)}")

    def save_reflectance(self):
        if self.sim_R_max is None or self.sim_lams is None:
            messagebox.showwarning("Advertencia", "No hay reflectancia óptima calculada.")
            return
        
        try:
            st.guardar_reflectancia(self.sim_lams, self.sim_R_max)
            messagebox.showinfo("Éxito", "Reflectancia óptima guardada correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar:\n{str(e)}")

    def compare_reflectances(self):
        f1 = self.entry_f1.get()
        f2 = self.entry_f2.get()
        if not f1 or not f2:
            messagebox.showwarning("Advertencia", "Debe seleccionar dos archivos para comparar.")
            return
        
        try:
            u1 = float(self.u1_var.get())
            u2 = float(self.u2_var.get())
            
            fig = st.comparar_reflectancias(archivo1=f1, archivo2=f2, archivo1_unidad=u1, archivo2_unidad=u2)
            
            if fig:
                self.embed_figures([fig], self.canvas_comp_frame, max_cols=1)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al comparar:\n{str(e)}")

if __name__ == "__main__":
    app = SimuladorARC()
    app.mainloop()
# %%
