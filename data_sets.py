from datasets import load_dataset
import json

# Cargar dataset
# dev = load_dataset('SWE-bench/SWE-bench', split='dev')
test = load_dataset('SWE-bench/SWE-bench', split='test')
# train = load_dataset('SWE-bench/SWE-bench', split='train')
# tot_len = len(dev) + len(test) + len(train)
print(f"Dataset cargado con {len(test)} entradas.")
# Convertir a lista de diccionarios
data = [item for item in test]

# Guardar en un archivo JSON
with open("swe_bench_lite_test.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Archivo guardado como swe_bench_lite_test.json")