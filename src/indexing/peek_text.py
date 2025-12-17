import os


def get_base_dir() -> str:
    # project root: two levels up from src/indexing
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, '..', '..'))


def read_doc_text(doc_id: str, corpus_dir: str) -> str:
    path = os.path.join(corpus_dir, doc_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Documento no encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    base_dir = get_base_dir()
    corpus_dir = os.path.join(base_dir, "data", "corpus1")

    print("=== Peek text from corpus1 document ===")
    print(f"Carpeta corpus: {corpus_dir}")

    doc_id = input("Ingresa el ID del documento (ej: 1, 000365356800057): ").strip()
    pos_str = input("Ingresa la posición (entero >= 0): ").strip()

    try:
        pos = int(pos_str)
        if pos < 0:
            raise ValueError
    except ValueError:
        print("Posición inválida. Debe ser un entero >= 0.")
        return

    try:
        content = read_doc_text(doc_id, corpus_dir)
    except Exception as e:
        print(f"Error leyendo documento: {e}")
        return

    if pos >= len(content):
        print(f"La posición {pos} está fuera de rango (longitud={len(content)}).")
        return

    snippet = content[pos:]
    print("\n--- Texto desde la posición ---")
    print(snippet)
    print("\n--- Fin ---")


if __name__ == "__main__":
    main()
