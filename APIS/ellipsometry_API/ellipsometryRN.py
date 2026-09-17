#%%
tmm_path = r"./tmm_core.py"
import sys
import importlib
import numpy as np
from tmm_utils_Rodrigo import load_fn, calculate_RT, cauchy_fn, brugg_fn, load_interp, stack2tmm, constant_fn
spec = importlib.util.spec_from_file_location("tmm", tmm_path)
tmm = importlib.util.module_from_spec(spec)
sys.modules["tmm"] = tmm 
spec.loader.exec_module(tmm)
#%%
ruta_materiales='./indices'
modelo_T1 = cauchy_fn(2.338,1.906,0.824)   #muestra T1 del paper (R-1)
modelo_TiO2 = cauchy_fn(2.274,2.781,7.359) #muestra numero 2 de TiO2 fabricada por sputtering
f_T1 = 0.58
materials = {
    'air': (constant_fn(1.0), constant_fn(0.0)),
    'SiO2': load_interp(f'{ruta_materiales}/nkdata/SiO2.nkv', skiprows=1),
    'Si': load_interp(f'{ruta_materiales}/nkdata/optical/Si.nk', skiprows=1),
}
for i in np.arange(0,1,0.1):
    i_round = round(i, 1)
    materials[f'TiO2_poroso_{i_round}'] = (brugg_fn(modelo_TiO2, constant_fn(1.0), i_round), load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows=1)[1])
    materials[f'SiO2_poroso_{i_round}'] = (brugg_fn(materials['SiO2'][0], constant_fn(1.0), i_round), materials['SiO2'][1])
#%%
# Primero una lista que va a tener todas las combinaciones de stacks posibles
# identificadas con un numero de id
stacks = []
# Reducimos los espesores para pruebas rápidas. Puedes cambiarlo a np.arange(10,200,10) si deseas el espacio completo.
espesores = np.arange(30, 150, 40)
n_o = 0
for i in materials.keys():
    for j in materials.keys():
        for k in espesores:
            for l in espesores:
                stacks.append({
                    'id': n_o,
                    'stack': [
                        [np.inf,'air','i'],
                        [k,i,'c'],
                        [l,j,'c'],
                        [np.inf,'Si','i']]})
                n_o += 1
#%%
curvas_psi_delta = []
curva_Is_Ic = []
lams = np.linspace(430,850,500)

X_list = []
y_list = []

print(f"Generando curvas elipsométricas para {len(stacks)} stacks...")
for id,stack in enumerate(stacks):
    if id % 1000 == 0 and id > 0:
        print(f"Procesados {id}/{len(stacks)} stacks...")
    n_list,d_list,c_list = stack2tmm(stack['stack'], materials, lams)
    ellips_res = tmm.ellips(n_list, d_list, np.radians(69.5), lams)
    psi = ellips_res['psi']
    delta = ellips_res['Delta']
    
    curvas_psi_delta.append((stack['id'],lams,psi,delta))
    Is_exp = np.sin(2 * psi) * np.sin(delta)
    Ic_exp = np.sin(2 * psi) * np.cos(delta)
    curva_Is_Ic.append((stack['id'],lams,Is_exp,Ic_exp))
    
    # Concatenar curvas Is e Ic como vector de características (input X)
    features = np.concatenate([Is_exp, Ic_exp])
    X_list.append(features)
    y_list.append(stack['id'])

X = np.array(X_list)
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
y = np.array(y_list)
#%%
# Ahora empiezo a formar la red neuronal
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)



# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        # Para CrossEntropyLoss (multiclase), las etiquetas y deben ser tipo Long (int64)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 64

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Check it's working
for batch, (X_batch, y_batch) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X_batch.shape}")
    print(f"y shape: {y_batch.shape}")
    break

#%%
# Definición de la Red Neuronal para clasificación multiclase
input_dim = X.shape[1]      # 1000 (Is y Ic concatenados)
hidden_dim = 256            # Dimensión oculta
output_dim = len(stacks)    # Número total de clases (stacks posibles)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        # Retornamos los logits sin Sigmoid ni Softmax, ya que CrossEntropyLoss lo calcula internamente
        x = self.layer_3(x)
        return x

# Instanciar el modelo
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
print("Estructura del modelo:")
print(model)

#%%
# Configurar entrenamiento
learning_rate = 0.01
# CrossEntropyLoss es la adecuada para clasificación multiclase en PyTorch
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 20
loss_values = []

print("Entrenando la red neuronal...")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        
        # Predicción y cálculo de pérdida
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        
        # Retropropagación
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    epoch_loss_avg = epoch_loss / len(train_dataloader)
    loss_values.append(epoch_loss_avg)
    print(f"Época {epoch+1}/{num_epochs} - Pérdida media: {epoch_loss_avg:.4f}")

print("¡Entrenamiento completado!")

# Graficar la evolución de la pérdida
fig, ax = plt.subplots(figsize=(8,5))
plt.plot(loss_values)
plt.title("Evolución de la Pérdida (Loss) por Época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

#%%
# Ejemplo de evaluación de una predicción
model.eval()
with torch.no_grad():
    # Tomamos una muestra de test
    X_test_sample, y_test_sample = test_data[0]
    # Hacer la predicción (añadiendo dimensión de batch)
    logits = model(X_test_sample.unsqueeze(0))
    # El índice con mayor valor (logit) es la predicción del modelo
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    print(f"Predicción del modelo (ID del stack): {predicted_class_id}")
    print(f"ID real del stack: {y_test_sample.item()}")
    
    # Mostrar la configuración del stack predicho
    stack_predicho = stacks[predicted_class_id]
    print(f"Configuración del stack predicho: {stack_predicho['stack']}")