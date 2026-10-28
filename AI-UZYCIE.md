Gemini 3.1 Pro: 
Działaj jako inżynier systemów niskopoziomowych. Wytłumacz mi architekturę i binarną strukturę formatu plików IDX, który jest wykorzystywany do zapisu wielowymiarowych macierzy w zbiorach takich jak Fashion-MNIST.
Jestem Senior Inżynierem Oprogramowania. Planuję napisać własny, wysokowydajny parser tego formatu w języku Rust, czytając dane z dysku. Nie potrzebuję ogólników, tylko twardej wiedzy o układzie bajtów.
Skup się na następujących aspektach technicznych:
Budowa nagłówka (Header): Wyjaśnij strukturę pierwszych kilkunastu bajtów. Czym jest Magic Number, jak zdekodować z niego typ bazowy (np. unsigned byte, float) oraz liczbę wymiarów (wymiarowość tensora).
Pułapka Endianness: Wyjaśnij dlaczego format IDX wymusza układ Big-Endian i jaki ma to wpływ na proces odczytu metadanych na współczesnych procesorach (x86/Apple Silicon), które działają w Little-Endian.
Układ Payloadu (Data Layout): Jak fizycznie na dysku ułożone są wielowymiarowe macierze (np. 60 000 obrazów, z których każdy ma wymiar 28x28). W jaki sposób wyliczyć offsety dla poszczególnych wierszy i obrazów.
Wskazówki I/O: Na co uważać projektując taki parser pod kątem systemowym, aby nie zadusić dysku i sprawnie zarządzać pamięcią.