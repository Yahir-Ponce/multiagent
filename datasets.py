from datasets import load_dataset
import json

# Cargar dataset
sbf = load_dataset('SWE-bench/SWE-bench_Lite', split="test")

# Convertir a lista de diccionarios
data = [item for item in sbf]

# Guardar en un archivo JSON
with open("swe_bench_lite_test.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Archivo guardado como swe_bench_lite_test.json")