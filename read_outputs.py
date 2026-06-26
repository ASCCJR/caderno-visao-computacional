import json
import sys
import os
import glob
import re

def clean_ansi(text):
    # Remove códigos de cor ANSI que o Colab gera no traceback
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def read_notebook(notebook_path):
    print("=" * 80)
    print(f"NOTEBOOK: {notebook_path}")
    print("=" * 80)
    try:
        with open(notebook_path, encoding="utf-8") as f:
            nb = json.load(f)

        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code" and cell.get("outputs"):
                has_output = False
                cell_out = []
                for output in cell["outputs"]:
                    otype = output.get("output_type", "")
                    if otype == "error":
                        cell_out.append(f"\n[ERRO] {output.get('ename', '')}: {output.get('evalue', '')}\n")
                        if "traceback" in output:
                            # Limpa os códigos ANSI para a IA ler melhor
                            trace = "".join(output["traceback"])
                            cell_out.append(clean_ansi(trace) + "\n")
                        has_output = True
                    elif "text" in output:
                        cell_out.append("".join(output["text"]))
                        has_output = True
                    elif "data" in output and "text/plain" in output["data"]:
                        cell_out.append("".join(output["data"]["text/plain"]))
                        has_output = True

                if has_output:
                    print(f"\n=== Célula {i} ===")
                    print("".join(cell_out).strip())
    except Exception as e:
        print(f"Erro ao ler {notebook_path}: {e}")

if __name__ == "__main__":
    # Garante suporte a UTF-8 no console do Windows para evitar erros ao imprimir setas ou emojis
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Em versões antigas do Python 3.x reconfigure pode não existir

    path_arg = sys.argv[1] if len(sys.argv) > 1 else None


    if path_arg is None:
        print("Usage: python read_outputs.py <notebook.ipynb | diretorio>")
        sys.exit(1)

    if os.path.isdir(path_arg):
        notebooks = glob.glob(os.path.join(path_arg, "**/*.ipynb"), recursive=True)
        if not notebooks:
            print(f"Nenhum arquivo .ipynb encontrado no diretório: {path_arg}")
        for nb in sorted(notebooks):
            read_notebook(nb)
            print("\n")
    else:
        read_notebook(path_arg)
