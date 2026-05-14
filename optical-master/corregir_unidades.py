#%%
nombre = "InGaPSch"
archivo = f'n/{nombre}'
with open(f'{archivo}.in3', 'r') as f:
    lineas = f.readlines()

with open(f'{nombre}_corregida.in3', 'w') as f:
    f.write(lineas[0]) # Título
    # Corregir línea de Rango
    rango = lineas[1].split()
    f.write(f"{float(rango[0])*10} {float(rango[1])*10}\n")
    
    for i in range(2, len(lineas)):
        partes = lineas[i].split()
        if len(partes) == 2: # Si es una fila de datos (lambda n/k)
            f.write(f"{float(partes[0])*10}\t{partes[1]}\n")
        else: # Si es la línea de conteo de puntos
            f.write(lineas[i])
#%%