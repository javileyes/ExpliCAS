---
trigger: always_on
---

En cada ciclo de desarrollo los agentes tienen que comprobar todos los tests y comprobar los benchmarks: si el ciclo de desarrollo ha sido de optimización los benchmarks deberían de mejorar signficativamente. Si no son de optimización y si ha habido refactorización para mejor mantenimiento o habido un aumento de funcionalidad (más reglas y funciones) entonces se permitirá un aumento moderado en benchmarks siempre que no sean críticos y estén justificados.
Para asegurar la correcta comparación de los benchmarks y comparar manzanas con manzanas, cada nuevo conjunto de test que se vaya creando se creará en un archivo a parte para evitar contaminación de los test anteriores y sus benchmarks. Un histórico de benchmarks se añadirán en un fichero de texto ./benchmarks.log, al final de cada benchmark se añadirá el identificador del último commit (y su comentario) para saber su temporización.