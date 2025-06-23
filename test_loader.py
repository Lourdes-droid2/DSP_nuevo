import sys
# Añadir el directorio raíz al path para poder importar los otros módulos
sys.path.append('.')

from load_signal import load_signal_from_wav

print("Intentando cargar 'p336_007.wav'...")
signal, fs = load_signal_from_wav("p336_007.wav", target_fs=48000)

if signal is not None:
    print(f"Carga exitosa. Frecuencia de muestreo: {fs} Hz. Longitud de la señal: {len(signal)} muestras.")
    # Verificar que la frecuencia de muestreo sea la esperada (la función ya lo hace, pero doble chequeo)
    if fs != 48000:
        print(f"ERROR: Se esperaba fs=48000, pero se obtuvo {fs}")
    else:
        print("Frecuencia de muestreo es correcta (48kHz).")
else:
    print("Fallo al cargar la señal.")

print("\nIntentando cargar con una frecuencia de muestreo incorrecta (esperando que falle)...")
signal_fail, fs_fail = load_signal_from_wav("p336_007.wav", target_fs=44100)

if signal_fail is None:
    print("Carga falló como se esperaba debido a fs incorrecta.")
else:
    print("ERROR: La carga no falló como se esperaba con fs incorrecta.")
