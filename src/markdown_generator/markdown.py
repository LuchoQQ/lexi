import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple

def parse_metadata_block(text: str) -> Dict[str, Any]:
    """
    Extrae los metadatos del bloque de metadatos.
    Los metadatos están entre guiones triples (---) y siguen el formato YAML.
    """
    metadata = {}
    pattern = r"^---\n(.*?)\n---"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        metadata_text = match.group(1)
        for line in metadata_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Manejar listas en formato YAML
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value.replace("'", "\""))
                    except:
                        value = [item.strip(' "\'') for item in value[1:-1].split(',')]
                
                # Remover comillas si están presentes
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                # Convertir "null" a None en Python
                elif value.lower() == "null":
                    value = None
                
                metadata[key] = value
    
    return metadata

def extract_references(text: str) -> List[str]:
    """
    Extrae las referencias marcadas con > al inicio de la línea, sin incluir el prefijo "Referencias".
    """
    references = []
    reference_pattern = r"^>\s*(.+)$"
    in_references_section = False
    
    for line in text.split('\n'):
        if line.strip().startswith("> **Referencias:**"):
            in_references_section = True
            continue
        
        match = re.match(reference_pattern, line)
        if match and in_references_section:
            references.append(match.group(1).strip())
        elif match:
            # Referencias fuera de la sección
            references.append(match.group(1).strip())
    
    return references

def clean_content(text: str) -> str:
    """
    Limpia el contenido eliminando metadatos y referencias.
    """
    # Eliminar bloques de metadatos
    text = re.sub(r"---\n.*?\n---\n?", "", text, flags=re.DOTALL)
    
    # Eliminar referencias (líneas que comienzan con >)
    lines = []
    for line in text.split('\n'):
        if not line.strip().startswith('>'):
            lines.append(line)
    
    # Eliminar líneas vacías al principio y al final
    result = '\n'.join(lines).strip()
    
    return result

def split_by_metadata_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Divide el texto por bloques de metadatos y extrae el contenido correspondiente a cada bloque.
    """
    # Dividir el texto por bloques de metadatos
    metadata_pattern = r"(---\n.*?\n---)"
    parts = re.split(metadata_pattern, text, flags=re.DOTALL)
    
    blocks = []
    current_metadata = {}
    current_content = ""
    
    for i, part in enumerate(parts):
        if part.startswith('---') and part.endswith('---'):
            # Es un bloque de metadatos
            if current_content or current_metadata:
                # Guardar el bloque anterior
                blocks.append({
                    "metadata": current_metadata,
                    "content": current_content.strip()
                })
            
            # Iniciar un nuevo bloque
            current_metadata = parse_metadata_block(part)
            current_content = ""
        else:
            # Es contenido
            current_content += part
    
    # Añadir el último bloque
    if current_content or current_metadata:
        blocks.append({
            "metadata": current_metadata,
            "content": current_content.strip()
        })
    
    return blocks

def process_markdown_file(file_path: str, output_path: Optional[str] = None) -> None:
    """
    Procesa un archivo Markdown, lo divide en chunks basados en bloques de metadatos
    y guarda el resultado en un archivo JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_text = file.read()
        
        # Dividir el texto en bloques basados en la metadata
        blocks = split_by_metadata_blocks(markdown_text)
        
        chunks = []
        
        for block in blocks:
            metadata = block["metadata"]
            content_with_refs = block["content"]
            
            # Extraer referencias
            references = extract_references(content_with_refs)
            
            # Limpiar contenido (eliminando también referencias)
            content = clean_content(content_with_refs)
            
            # Identificar encabezados en el contenido
            header_match = re.search(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
            header = None
            header_level = None
            
            if header_match:
                header_level = len(header_match.group(1))
                header = header_match.group(2).strip()
            
            # Crear el chunk
            chunk = {
                "content": content,
                "metadata": metadata,
                "references": references
            }
            
            # Añadir información del encabezado si está disponible
            if header:
                chunk["header"] = header
                chunk["header_level"] = header_level
            
            chunks.append(chunk)
        
        # Si no se especifica ruta de salida, usar el nombre del archivo original
        if output_path is None:
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}_chunks.json"
        
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(chunks, json_file, ensure_ascii=False, indent=2)
            
        print(f"Se han generado {len(chunks)} chunks y guardado en {output_path}")
        
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")

# Ejemplo de uso
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        process_markdown_file(input_file, output_file)
    else:
        print("Uso: python script.py archivo_markdown.md [archivo_salida.json]")
        print("Si no se especifica archivo de salida, se usará el nombre del archivo de entrada con _chunks.json")