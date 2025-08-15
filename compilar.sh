#!/bin/bash

# Nombre del archivo fuente
SRC="peces33.cpp"
# Nombre del ejecutable
OUT="peces33"

# Detectar flags de OpenCV automáticamente
OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)

# Compilar con g++, habilitando OpenMP (-fopenmp)
g++ -O3 -fopenmp -std=c++17 "$SRC" -o "$OUT" $OPENCV_FLAGS

# Comprobar si la compilación fue exitosa
if [ $? -eq 0 ]; then
    echo "✅ Compilación completada: ./$OUT"
else
    echo "❌ Error de compilación"
fi

