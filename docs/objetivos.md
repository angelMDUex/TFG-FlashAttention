# Objetivos
Los objetivos de este Trabajo de Fin de Grado son:

O1- Implementar en CUDA el algoritmo FlashAttention no causal para una longitud de secuencia y dimension de cabeza determinada.
02- Justificar las decisiones tomadas durante la implementacion de O1.
O3- Implementar en Triton una version equivalente a la implementacion CUDA del objetivo O1.
O4- Comparar las diferencias de rendimiento entre las versiones CUDA, Triton, Torch SDPA y Cuddn SDPA y dar una explicacion a los distintos resultados obtenidos.

Una implementacion de alto rendimiento en cuda es dificil de leer, implementar y en ocasiones dificil de explicar. Una implementacion triton, por el contrario, es mas sencilla de leer, sencilla de escribir, y cambios de parametros que, en cuda requeririan reescribir gran parte del kernel, en triton seria un simple cambio de numeros.

Sin embargo, dificil, sencillo, simple, son adjetivos que no se pueden cuantificar de manera objetiva. Describiendo todas las optimizaciones aplicadas en cuda, la ingenieria detras de las decisiones tomadas y leyendo el codigo resultante, el lector podra decidir que opcion prefiere.

