from pathlib import Path

def contar_imagenes(ruta_principal):
    # Definimos las extensiones de imagen que queremos contar
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}
    directorio = Path(ruta_principal)
    
    if not directorio.exists() or not directorio.is_dir():
        print("La ruta especificada no existe o no es una carpeta.")
        return

    print(f"Analizando: {directorio.resolve()}\n" + "-"*40)

    # Iteramos solo sobre las subcarpetas de primer nivel
    for subcarpeta in directorio.iterdir():
        if subcarpeta.is_dir():
            contador = 0
            
            # rglob('*') busca de forma recursiva todo el contenido de la subcarpeta
            for archivo in subcarpeta.rglob('*'):
                # Comprobamos que sea un archivo y que su extensión (en minúsculas) sea válida
                if archivo.is_file() and archivo.suffix.lower() in extensiones_validas:
                    contador += 1
                    
            print(f"📁 {subcarpeta.name}: {contador} imágenes")

# --- USO ---
# Sustituye esta ruta por la de tu carpeta principal
ruta_base = "data/val" 
contar_imagenes(ruta_base)