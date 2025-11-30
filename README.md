# Symulator Fizyki Kulek 3D

**Autor:** Mateusz Siwy

## Opis

Symulacja fizyki kulek w obracającym się sześcianie 3D z wykorzystaniem OpenGL, shaderów GLSL i oświetlenia Phonga.

### Funkcjonalność
- Realistyczna fizyka (grawitacja, kolizje, odbicia)
- Obracający się sześcian z transformacją grawitacj
- Shadery GLSL z oświetleniem Phonga
- Dynamiczne źródła światła (świecące kulki)
- Tekstury proceduralne z mipmap
- Interaktywna kamera (LPM + scroll)
- Cienie

## Uruchomienie

```bash
pip install -r requirements.txt
python main_cube.py
```

## Sterowanie

- **LPM + przeciągnięcie** - obrót kamery
- **Scroll myszy** - zoom
- **Strzałki ←/→** - obrót sześcianu
- **Spacja** - pauza/wznowienie
- **R** - reset
- **L** - dodanie świecącej się kulki
- **+** - dodanie normalnej kulki

## Użyte biblioteki

- **PyQt6** - interfejs GUI
- **PyOpenGL** - bindingi OpenGL
- **NumPy** - obliczenia matematyczne (macierze, fizyka)
- **Pillow** - ładowanie tekstur PNG

## Assety

- `shaders/` - shadery GLSL (vertex + fragment)
- `textures/` - 7 tekstur proceduralnych PNG

## Filmik

https://drive.google.com/file/d/1AuwaSxJqS-aSguFXmA64qk9rqVK8oa0E/view?usp=sharing
---

Projekt edukacyjny | Listopad 2025
