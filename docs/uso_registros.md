# Algoritmo general FlashAttention-2.
Sean las matrices

$$
Q,K,V \in \mathbb{R}^{N\times d},
$$

y sean \(B_r\) y \(B_c\) los tamaños de bloque empleados para \(Q\) y para \(K,V\), respectivamente.

La matriz \(Q\) se divide en

$$
T_r=\frac{N}{B_r}
$$

bloques

$$
Q_1,\ldots,Q_{T_r},
\qquad
Q_i\in\mathbb{R}^{B_r\times d}.
$$

De forma análoga, \(K\) y \(V\) se dividen en

$$
T_c=\frac{N}{B_c}
$$

bloques

$$
K_1,\ldots,K_{T_c},
\qquad
V_1,\ldots,V_{T_c},
$$

con

$$
K_j,V_j\in\mathbb{R}^{B_c\times d}.
$$

Para cada bloque \(Q_i\):

1. Cargar \(Q_i\) desde HBM a memoria on-chip.

2. Inicializar:

$$
O_i^{(0)}=0,
$$

$$
l_i^{(0)}=0,
$$

$$
m_i^{(0)}=-\infty.
$$

Aquí, \(O_i\) representa el acumulador de la salida, \(m_i\) almacena el máximo actual de cada fila y \(l_i\) almacena el denominador acumulado del softmax.

3. Para cada bloque \(K_j,V_j\):

   a. Cargar \(K_j\) y \(V_j\) desde HBM a memoria on-chip.

   b. Calcular la matriz de scores:

$$
S_i^{(j)}
=
Q_iK_j^T,
\qquad
S_i^{(j)}\in\mathbb{R}^{B_r\times B_c}.
$$

c. Actualizar el máximo de cada fila:

$$
m_i^{(j)}
=
\max
\left(
m_i^{(j-1)},
\operatorname{rowmax}(S_i^{(j)})
\right).
$$

d. Calcular el factor necesario para reescalar los resultados obtenidos en iteraciones anteriores:

$$
\alpha_i^{(j)}
=
\exp
\left(
m_i^{(j-1)}-m_i^{(j)}
\right).
$$

e. Calcular las exponenciales correspondientes al bloque actual:

$$
P_i^{(j)}
=
\exp
\left(
S_i^{(j)}-m_i^{(j)}
\right).
$$

La resta de \(m_i^{(j)}\) se realiza de forma independiente para cada fila de \(S_i^{(j)}\).

f. Actualizar el denominador acumulado del softmax:

$$
l_i^{(j)}
=
\alpha_i^{(j)}\,l_i^{(j-1)}
+
\operatorname{rowsum}
\left(
P_i^{(j)}
\right).
$$

g. Actualizar el acumulador de salida:

$$
O_i^{(j)}
=
\alpha_i^{(j)}\,O_i^{(j-1)}
+
P_i^{(j)}V_j.
$$

4. Una vez procesados todos los bloques \(K_j,V_j\), normalizar cada fila del acumulador mediante su denominador correspondiente:

$$
O_i[r,:]
=
\frac{
O_i^{(T_c)}[r,:]
}{
l_i^{(T_c)}[r]
},
\qquad
0\le r < B_r.
$$

5. Escribir \(O_i\) en HBM como el bloque correspondiente de la matriz de salida \(O\).

El procedimiento se repite para todos los bloques \(Q_i\), obteniendo finalmente

$$
O=\operatorname{softmax}(QK^T)V.
$$

# Calculo de ocupancia y mapeo de warps.
La ocupancia es el numero de threads que se pueden asignar a un SM sobre el total. En un SM caben 16 ctas, cada uno con 1024 threads maximo, haciendo un total de 1536 threads (48 warps). La ocupancia tambien se ve afectada por la sram alojada por cada cta. 

Una mayor cantidad de warps asignadas a una sm permite intercalar las operaciones entre los mismos. Si un warp ejecuta una operacion costosa y tiene una dependencia de datos, el streaming multiprocesor continua ejecutando esa operacion en el background mientras un nuevo warp, con una nueva instruccion, es encolado, consiguiendo una paralelismo a nivel de instrucciones. 

Sin embargo, es dificil saber con que ocupancia rinde mejor un kernel. Puede ser que el kernel sea memory bound y que se beneficie de mayor ocupancia para enmascarar las latencias, puede ser tambien que un unico warp sature ya los pipelines que actualmente formen el cuello de botella del kernel (como los tensor cores) y que no compense aumentar el numero de warps (o que simplemente si lo haga pero que no tengamos una manera cierta de cuantificar cuanto mejor va a ser nuestro kernel). Si aumentamos el numero de warps por cta, pero reducimos el % de warps activos por ciclo, o el tiempo medio de espera en barreras sea mayor, entonces no compensa una mayor ocupancia. Si reducimos la ocupancia pero a cambio necesitamos mas accesos a smem porque no podemos reutilizar los datos, a lo mejor perdemos rendimiento. Que el numero de registros sea un factor limitante y que haya que acceder a la memoria local para conseguir una mayor ocupancia. Esos accesos a memoria pueden compensar a la ocupancia, pero tambien perjudicar el rendimiento.

A continuacion una serie de tradeoffs para con la ocupancia.

Reducir registros mediante recomputo. Es de las mas inocuas al rendimiento de una aplicacion. El warp recalcula una operacion multiples veces para favorecer que la latencia de operaciones mas caras se compensen con los warps nuevos. 

Reducir los registros mediante un particionado de trabajo. Es dificil de calibrar. Si un warp realiza cierta cantidad de trabajo y podemos reducirla sin aumentar la sincronizacion entre los threads, entonces la cantidad de registros se reduce proporcionalmente y obtenemos nuevos niveles de paralelismo. Si el warp realiza cierta cantidad de trabajo y podemos reducirla pero a costa de aumentar la sincronizacion entre los warps, entonces la pregunta es si ese nivel anadido de paralelismo compensa la sincronizacion, ganaremos rendimiento, si no, lo perderemos aun habiendo aumentado la ocupancia y el paralelismo. 

Forzar spills a dram. Es de las menos recomendables. Fuerza que los registros sobrantes para conseguir la ocupancia sean volcados a memoria principal y sean recuperados de la misma cada vez que se usen. En pocas ocasiones suele pasar que un kernel aumente el rendimiento si aumenta la ocupancia a costa de dram spills.

Forzar spills a sram. Es una opcion relativamente novedosa (Cuda 13.0). Los registros se vuelcan a sram en vez de a dram, que tiene menor latencia. 

Aumentar el numero de warps/cta. Puede ser beneficiosa porque puede reutilizar los datos de sram a nivel de bloque. Si no hay __syncthreads(), entonces el beneficio es practicamente gratuito. Si hay __syncthreads(), entonces puede reducirse el rendimiento. Cuantos mas warps llegan a la barrera, menos hay para ejecutar. Si se compensa el numero de warps aumentando el numero de ctas, menos coste tiene la sincronizacion de los mismos, a costa de requerir mas sram. Mas ctas por sm en vez de mas warps por sm tambien realizan mas accesos a memoria.

Yo calculo el numero de registros/smem por warp/cta respectivamente, con un grafo de ejecucion que tenga sentido, y en caso de requerir mas ocupancia aumentar el numero de warps/cta pudiendo emplear sram spills. 

# Uso de registros y SRAM.
El uso de registros y sram del kernel por los componentes principales es el siguiente:

El tile Q_i e R{16x128} del tipo bfloat16: 
16 x 128 elementos * 1 registro/2 bfloat16  = 1024 registros.
1024 registros / 32 threads/warp = 32 registros/thread.
El tile Q se carga directamente desde la memoria principal a registros.

Los tiles K_i y V_i e R{16x128} del tipo bfloat16:
Los tiles K_i y V_i se cargan en sram y a continuacion en registros. 
2x16x128 bfloat16 * 2bytes/bfloat16 = 8192 bytes en sram.
El uso de ldmatrix.x4 carga dos tiles de 16x8x2.
El doble buffer sram-registros requiere 16x8x2x2 elementos de espacio en registros.
16x8x2x2 elementos * 1 registro / 2bfloat16 = 256 registros.
256 registros * 1 warp / 32 threads = 8 registros / thread.
En la practica, cuando terminamos de procesar K, V puede reutilizar ese mismo espacio en registros.

La matriz de scores S_i, resultado de Q_i * K_i:
16x16 fp32 * 1 fp32/ 1 registro = 8 registros thread.

La matriz P_ij:
16x16 fp32 * 1 fp32/ 2bfloat = 4 registros thread.
En la practica, se deben reusar los registros de S.

El maximo por row actual:
Dado el formato de A e R{16x16}, cada thread almacena 4 registros correspondientes a 4x8x8 tile.
El maximo por row y por thread ocupa entonces 2 registros.

El maximo por row anterior:
Es identico al maximo por row y thread, que son 2 registros.

El row denominator es:
2 registros

El row denominator anterior es:
2 registros

El output accumulator es:
16x128 elementos fp32 * 1 warp / 32 threads = 64 registros.

El uso esperado de los registros es:
Mantener Q, Output accumulator, Maximo por row, Maximo por row anterior, row denominator, row denominator anterior, tiles de K/V y matriz S en registros.
La matriz P se espera que utilice registros de S. La matriz S consume registros adicionales porque no puede utilizar los de output accumulator. 

Q_i                 32
O_i                 64
S_ij                  8
K/V_i,i+i              8
m_i / m_prev          2
l_i / l_prev          2
scale_factor           2
row_sum                2
------------------------
TOTAL                120 regs/thread

La cifra de 120 registros se interpreta de la siguiente manera:
El compilador puede decidir optimizar operaciones y almacenar el resultado en registros, aumentando la cuenta de registros empleados. Este tipo de operaciones pueden ser de indexacion, por ejemplo. El compilador tambien puede decidir sacrificar registros para conseguir una mayor ocupancia. Si se utilizan menos de 120 registros, datos que son reutilizados multiples veces, como O_i, Q_i, deben ser almacenados en memoria local y recargados cuando se usen. Es pronto para afirmar que una cuenta de registros menor a 120 aumente o disminuya el rendimiento. Menor cantidad de registros implican mas operaciones de memoria, pero mayor ocupacion -La SM puede intercalar las operaciones extra de memoria con mas operaciones fruto de tener mas warps en la maquina-. Mas registros pueden reducir el numero de warps que residen en un SM, y con ello reducir la variedad de operaciones, pero disminuyen el numero de accesos a memoria principal.

# Decisiones de paralelismo y warp coarsening.
Se paraleliza a traves de la longitud de secuencia. Paralelizar a traves de la dimension de cabeza requiere trabajo desigual por parte de los threads durante el calculo de la matriz P, ademas de forzar sincronizacion de la matriz P y de las sumas de Q_j * K_j. Decido que el numero de rows que procesa un warp sea 16, porque es la dimension que tiene la matriz A(m16k16) de mma.sync. Procesar un multiplo de esas 16 filas multiplica el numero de registros por warp y reduce el nivel de paralelismo. Aumentar el nivel de paralelismo implica que si el hardware mejora, aumenta el ancho de banda de la dram, y aumentan el numero de SMs, se puede aprovechar mas. 

# Cargado Q_i: 
Cargar Q_i presenta numerosas opciones. Q_i multiplica cada K_i y V_i. Es logico que el factor que mas hay que cargar de memria se almacene en la memoria de mayor velocidad, es decir, los registros. Hay multiples maneras de cargar Q_i, cada una con sus ventajas e inconvenientes.

La manera mas directa de cargar Q_i es la siguiente:
Dividir Q_i en Q_ij tiles de 16x16 elementos. Dentro de cada warp, los lanes se dividen en grupos de 4 lanes para formar 8 grupos de 4 lanes cada uno. Cada lane carga entonces 2 bf16 por registro por cuadrante de 8x8. El orden es el siguiente:

Acceder de esta manera a memoria principal no es recomendable. Los elementos de cada columna estan separados por 2 bytes/ 1bf16 * 128 bf16 = 256 bytes. Acceder a 8 rows carga 8 lineas de cache de 128 bytes, de los cuales solo se cargan sectores de 16 bytes por cuadrante de 8x8. Es decir, 16 bytes / 128 bytes son utilizados. Se puede pensar que la linea de cache sigue en memoria al cargar el siguiente bloque de 8x8 o 16x16, como es probable que ocurra en CPU, pero sin conocer la politica de reemplazo ni conocer la contencion de la misma (es decir, cuantos mas warps acceden a memoria menos probable es que un dato se mantenga en la cache), no se puede afirmar con seguridad. Ademas, Q es un factor local a cada warp; no se reutiliza entre warps. Es funcional, aun asi. 

La segunda manera de cargar Q es en SRAM y de sram a RF. 
El espacio total de Q es 4096 bytes (16x128 bfloat * 2 bytes/bloat16). La sram se comparte por todos los warp schedulers, 4 en sm_8X. NCU define un wavefront como la maxima cantidad de bytes que puede leer un warp de la sram, suponiendo que no hay conflictos de banco. Con 32 bancos * 4 bytes/banco, 128 bytes constituyen un wavefront. Las preguntas sin resolver son las siguientes:
1- Cual es la latencia L2-sram y cual es la latencia sram-rf? Sin una tabla de latencias, no podemos afirmar que un acceso L2-sram + sram-rf es mas barato que un acceso L2-L1, y de L1 a RF.
2- Si dos warps encolados en diferentes processing elements acceden a la sram simultaneamente, los accesos se serializan? Nvidia define conflictos de banco a nivel de warp, pero no explica que ocurre si hay accesos concurrentes a la sram. 
3- Si cargamos Q en memoria compartida, entonces, el cta pasa a tener (suponiendo un doble buffer de  K_i, V_i) como minimo 24Kb. No podemos reutilizar los bufferes de K_i/V_i porque entonces secuencializamos las cargas (No podemos cargar K en el espacio asignado a Q). Para una matriz que solo se va a cargar una vez, perdemos 4kb/cta durante todas la ejecucion del cta, reduciendo la ocupancia. 
4- No sabemos el throughput de la L1. Con la sram sabemos que como maximo podemos cargar 128 bytes * warp/ciclo. Con la l1 no. Si utilizamos cargas vectorizadas, la l1 puede proporcionar los 16 bytes / thread * 32 thread/warp = 512 bytes ciclo? O se dividen en micro instrucciones y el throughput se divide a la mitad (256 bytes) o a un cuarto (128 bytes)? Si el throughput es de 128 bytes/ciclo, entonces, por que se existen instrucciones de cargas vectorizadas?
5- No sabemos la latencia l2-l1 l2-rf frente a l2-rf (uncached loads).

Si estamos utilizando la memoria sram para K y V, entonces la L1 queda practicamente libre (salvo para carga de instrucciones). Cuantos mas ctas se encolen en una sm, mayor sera la contencion de la sram. Aliviar la carga utilizando la L1 no es una idea descabellada. La pregunta principal es si con las instrucciones extra que conlleva la permutacion mariposa podemos superar el rendimiento de la sram. Nos faltan datos para poder responder a la pregunta de manera objetiva, pero yo me he inclinado a utilizar la l1. Pienso que la l1 opera en paralelo a la sram, y que la sram puede ser victima de contencion, ademas de limitar el paralelismo entre cargar Q y K/V y/o de reducir la ocupancia.

Finalmente, se instruye al compilador para que las cargas de Q tengan la modalidad streaming. Ademas, se hace un bypass de la L1.

# El patron mariposa:
32 threads / warp * 16 bytes / 1 thread = 512 bytes/warp por instruccion de carga vectorizada.
1 * 128 bf16 * 2 elementos/bf16 = 256 bytes.
Cada grupo de 16 lanes puede cargar un row de Q_i utilizando cargas vectorizadas de 16 bytes.
Cada lane se presenta con 64 registros de 32 bits. Para cada lane_i, el registro j es el elemento que deberia tener el lane_j en la posicion i. Es decir, una traspuesta en registros. 

# Instrucciones predicadas.
Warp divergence ejecuta ambas partes del salto y materializa, para cada lane, los registros correspondientes al path que ejecutan. Instrucciones predicadas son instrucciones que materializan su resultado si un cierto predicado es cierto a nivel de lane. La principal diferencia es que las instrucciones predicadas pueden eliminar un salto a costa de que un warp ejecute ambos paths, nvidia dice:

The compiler replaces a branch instruction with predicated instructions only if the number of instructions controlled by the branch condition is less than or equal to a certain threshold.
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html


# Instruccion ldmatrix.x4: Carga de un tile 16x16 de sram con layout mma.sync.
Esta instruccion nos permite cargar un tile 16x16 en registros desde la sram. Esta instruccion, frente a la instruccion ldmatrix.x2, que carga un tile 16x8, permite cargar dos de los mismos. Logicamente debe tener una latencia mayor, pero debe ser menor que dos ldmatrix.x2 issued por separado. 

# Instruccion cp.async: 
La instruccion cp.async permite al hardware transferir datos desde memoria principal a L2, y de L2 a sram sin pasar por registros. Comunmente esta instruccion se confunde el significado de async. Async no quiere decir "asincrono" en el sentido de que el warp pueda progresar mientras la carga sigue en el background, las instrucciones ld/st comunes ya tienen este efecto. Nvidia define que un cambio de warp ocurre cuando se detecta una dependencia no resuelta. Si nosotros lanzamos una ld y no usamos el resultado hasta cierta cantidad de instrucciones mas tarde, la carga sigue procesando y el warp sigue procesando. Las verdaderas ventajas de la instruccion cp.async son que no hace un consumo extra de registros y que no atraviesa l1 (.cg). Hay un bypass completo del register file.

# Doble buffer smem-rf:
Nvidia documenta que es posible realizar un doble buffer smem-rf. En el kernel cuda se establece un doble buffer de 2 tiles de 16x16.

# K/V y sram:
Mientras que Q_i se utiliza unicamente por warp, K_i y V_i se reutilizan por todos los warps. Si tenemos mas warps por cta, cargar K_i y V_i en sram permite reducir los accesos a l2 de manera proporcional al numero de warps. 

Actualmente, un buffer lo forma K_i y V_i. K_i y V_i forman ambos 8k de memoria. Se cargan cooperativamente entre los warps, que es un parametro determinado por optimizacion. 

## Sram layout naive:
Un layout naive de sram luce de la siguiente manera. 

Este layout ocasiona conflictos de banco. Un conflicto de banco ocurre cuando dos threads de un mismo warp acceden a -distintas- direcciones de memoria almacenadas en el mismo banco. Esto ocurra que los accesos se serialicen. NCU define wavefront como la maxima cantidad de bytes que un warp puede transferir a/desde la memoria sram. Actualmente son 128. Si 32 threads de un warp acceden a datos distintos de un mismo banco, por ejemplo, 4 bytes, la sram solo es capaz de suministrar 4 bytes/ciclo/thread. Si tarda 32 ciclos, solo 4 * 32 / 128 * 32, 4/128, 1/32 ancho de banda aprovechado.

Por tanto, los accesos a sram se tiene que procurar que sea a nivel de banco. 

## Sram layout padded:

Anadir un padding de tamano equivalente al dato que queramos acceder mitiga los conflictos de banco, pero perdemos memoria sram.


## Sram layout swizzled:
Un padding swizzled sacrifica complejidad de codigo para eliminar completamente los conflictos de banco. 


# Doble buffer sram:
Los warps cooperan para cargar los elementos de un multibuffer. La pregunta principal de los multiples bufferes es: Que tamano tiene que tener el doble buffer para maximizar la actividad del consumidor y la actividad del productor. Un buffer demasiado grande puede favorecer al consumidor, pero puede mantener en idle al productor por mas tiempo, un buffer demasiado pequeno perjudica al consumidor (hay mas burbujas), y al productor (quizas no se aprovecha el throughput). Un numero distinto de bufferes mayor a dos incluye mas sincronizacion entre los warps, pero ayuda a balancear la carga productor consumidor.

Actualmente, K_i y V_i forman bufferes distintos para cada i, y un cta puede cargar multiples. El numero de bufferes que maximizan productor-consumidor se determina mediante una optimizacion de parametros en el ultimo capitulo.

# Loop unrolling y partial loop unrolling:
Loop unrolling es una tecnica que convierte un bucle for de n iteraciones en una secuencia de instrucciones repetidas n veces. Las ventajas son: reducir o eliminar completamente los saltos como consecuencia del bucle y reorganizar las instrucciones pertenecientes originalmente a distintas iteraciones para aprovechar mejor las pipelines. El problema es que un loop unrolling excesivo puede producir fallos en la memoria cache de instrucciones, eliminando la ventaja de rendimiento. Cuanto desenrollar un bucle es una tarea de optimizacion.

Loop unrolling completo tambien ayuda a cuda a identificar variables de tipo arreglo como registros, siempre y cuando se accedan mediante constantes.

# Loop Counters Signed vs. Unsigned
Nvidia recomienda el uso de enteros con signo durante los bucles. Un entero sin signo tiene por regla volver a 0 en caso de overflow. Un entero con signo no esta definido por el standard del lenguaje, lo que permite al compilador a usar optimizaciones mas agresivas. Un ejemplo es la reduccion de fuerza. 

El compilador puede trasformar la multiplicacion en una suma por una constante. 

# Calcular el row-sum y el row-max
Estas operaciones necesitan calcular el maximo y la suma de datos que estan en registros de distintos threads del warp. Para calcular la suma, se utiliza el operador suma, y para el maximo la funcion fmaf.
Para acceder a registros del mismo warp, la instruccion shfl_xor_sync con width 4, para agrupar los threads en grupos de 4 lanes.

# Calculo de exponencial.
La exponencial es una operacion tradicionalmente cara de computar. Tanto es asi, que, si bien el throughput de las operacines matriciales ha seguido aumentando en hopper y blackwell, el throughput de las trascendentales se ha mantenido con respecto a ampere. 16 resultados/cyclo/sm es una cifra menor comparada con los 128 resultados/cyclo/sm de los pipeline fp32 en sm_86. En blackwell, estos 16 resultados/cyclo/sm se mantienen, lo que ha ocasionado tener que buscar aproximaciones para reducir este cuello de botella. Considero que en ampere tambien podemos hacer una aproximacion de expf con una precision aceptable. 

La opcion que he utilizado es:
1- Fusionar el scale con la exponencial
maxi(scale * sij)

como scale = 1/sqrt(d) > 0

se cumple que maxi(scale *si) = scale * maxi(si)

asi que podemos hacer scale * si - scale * m = scale(si - m)

Con eso podemos hacer x = score - row_max sin escalar ninguno de los dos, y despues:

exp_poly2_scaled(x), donde calculamos e^scale(score-row_max)

2- Calcular e^scale(score-row_max)
Usando la identidad a^b=2^blog2​(a),
Para a = e:
e^z = 2^zlog2(e)

Tomando z = scale * x, donde x = (score - row_max).

queda e^(scale * x) = 2^(scale * x)*logx(e)

definimos K = scale * log2(e), que es constante y puede calcularse en tiempo de compilacion

definimos y = x * K.

el problema se reduce a 2 ^ y.

3- Reduccion de rango.
Hasta ahora: e^z = 2^y,
y = n + f, donde n es round(y). y f e [-0.5,0.5]
e^z = 2^n*2^f

La funcion 2^f es una funcion suave en un intervalo pequeno (2
−0.5
≈0.7071
2
0.5
≈1.4142) que podemos aproximar con un polinomio.

4- Scollya
Scollya utiliza el algoritmo de remez para aproximar ese polinomio. La diferencia con respecto a un polinomio de taylor es que taylor aproxima muy bien cerca de un punto determinado, pero conforme aumenta la distancia, aumenta el error. El algorimo de remez minimiza el maximo error de la funcion en un punto. 

Mi resultado, tras 200 iteraciones es
2^f aprox = 1 + f * (0.702941834926605224609375 + x * 0.23986406624317169189453125)

con un error relativo de 0.19634037~3/4~e-2 

4.1 Error relativo


5- Calcular 2^n 
Calcular 2^n, dado el estandar ieee754, es simplemente una shift a la izquierda. 
La formula es: n + 127 << 23

5.1 Constante MAGIC


Sollya no necesita una semilla. 

## Analisis SASS __expf vs aproximacion por polinomio.
[Godbolt analisis](https://godbolt.org/#g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,selection:(endColumn:2,endLineNumber:56,positionColumn:2,positionLineNumber:56,selectionStartColumn:2,selectionStartLineNumber:56,startColumn:2,startLineNumber:56),source:'%23include+%3Ccuda_runtime.h%3E%0A%23include+%3Ccstdint%3E%0A%23include+%3Ccmath%3E%0A%0A%0Aconstexpr+uint32_t+SCALE_BITS+%3D+0x3db504f3u%3B++//+1/sqrt(128)%0A%0A%0Atemplate+%3Cuint32_t+scale_bits%3E%0A__device__+__forceinline__%0Afloat+exp_poly2_scaled(float+x)%0A%7B%0A++++constexpr+float+LOG2E+%3D+0x1.715476p%2B0f%3B%0A++++constexpr+float+MAGIC+%3D+0x1.8p23f%3B%0A%0A++++const+float+scale+%3D+__uint_as_float(scale_bits)%3B%0A++++const+float+K+%3D+scale+*+LOG2E%3B%0A%0A++++x+%3D+fmaxf(x,+-80.0f+/+scale)%3B%0A%0A++++float+t+%3D+fmaf(x,+K,+MAGIC)%3B%0A%0A++++uint32_t+nbits+%3D+__float_as_uint(t)%3B%0A%0A++++float+n+%3D+t+-+MAGIC%3B%0A%0A++++float+f+%3D+fmaf(x,+K,+-n)%3B%0A%0A++++float+p+%3D+fmaf(%0A++++++++0.23986406624317169189453125f,%0A++++++++f,%0A++++++++0.702941834926605224609375f%0A++++)%3B%0A%0A++++p+%3D+fmaf(p,+f,+1.0f)%3B%0A%0A++++return+__uint_as_float(%0A++++++++__float_as_uint(p)+%2B+(nbits+%3C%3C+23)%0A++++)%3B%0A%7D%0A%0A%0Aextern+%22C%22+__global__%0Avoid+test_expf(float*+out,+float+x)%0A%7B%0A++++constexpr+float+scale+%3D+0.0883883461356163f%3B%0A%0A++++out%5B0%5D+%3D+__expf(x+*+scale)%3B%0A%7D%0A%0A%0Aextern+%22C%22+__global__%0Avoid+test_poly2(float*+out,+float+x)%0A%7B%0A++++out%5B0%5D+%3D+exp_poly2_scaled%3CSCALE_BITS%3E(x)%3B%0A%7D'),l:'5',n:'0',o:'CUDA+C%2B%2B+source+%231',t:'0'),(h:compiler,i:(compiler:nvcc130,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:cuda,libs:!(),options:'-arch%3Dsm_86+-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+NVCC+13.0.0+(Editor+%231)',t:'0'),(h:device,i:(compilerName:'NVCC+13.0.0',device:'SASS+(sm_86)',editorid:1,fontScale:14,fontUsePx:'0',j:1,selection:(endColumn:7,endLineNumber:17,positionColumn:7,positionLineNumber:17,selectionStartColumn:7,selectionStartLineNumber:17,startColumn:7,startLineNumber:17),treeid:0),l:'5',n:'0',o:'Device+Viewer+NVCC+13.0.0+(Editor+%231,+Compiler+%231)',t:'0')),k:100,l:'4',n:'0',o:'',s:0,t:'0')),version:4)

La descripcion de las instrucciones es:
| Instruccion                              | Significado                                 | Throughput |
|------------------------------------------|---------------------------------------------|------------|
| MOV R3, 0x3e0293ee                       | mover a R3 el resultado de scale * log2(e)  | -          |
| FMNMX R0, R0, -905.0966796875, !PT       | clamp el resultado (fmaxf -80.0f / scale)   | 64         |
| MOV R2, 0x3e759eed                       | Cargar el coeficiente 0.2398...             | -          |
| FFMA R4, R0, R3, 12582912                | Fma con K y MAGIC                           | 128        |
| FADD R3, R4, -12582912                   | n = t - MAGIC                               | 128        |
| FFMA R3, R0, 0.12751743197441101074, -R3 | f = fmaf(x, K, -n);                         | 128        |
| FFMA R0, R3, R2, 0.70294183492660522461  | fmaf (0.23... , f, 0.70...)                 | 128        |
| FFMA R5, R3, R0, 1                       | Ultimo fma con p y f (+1 del coeficiente 1) | 128        |
| LEA R5, R4, R5, 0x17                     | p + nbits << 23                             | -          |

Son 4 fma, 1 max, 1 fadd, 2 operaciones de carga con inmediatos y una lea. 

Tpoly​≃4/128​+1/128​+1/64​=7/128​=0.0547

__expf
Calcula ez=2zlog2​(e).

| Instruccion                         | Significado                                   | Throughput |
|-------------------------------------|-----------------------------------------------|------------|
| FMUL R0, R0, 0.08838834613561630249 | El escalado de x por 1/sqrt(128)              | 128        |
| FMUL R0, R0, 1.4426950216293334961  | Multiplicar R0 por log2(e)                    | 128        |
| FSETP.GEU.AND P0, PT, R0, -126, PT  | Registro de predicado para clamp si x <= -126 | -          |
| @!P0 FMUL R0, R0, 0.5               | Si x <= -126, se divide entre 2               | 128        |
| MUFU.EX2 R5, R0                     | Calculo de 2^y                                | 16         |
| @!P0 FMUL R5, R5, R5                | Si x <= -126, se eleva al cuadrado R5         | 128        |

TEX2​=1/16​=0.0625

Sin contar las mov y las lea (que dificilmente pueden tener menos throughput que las fmul), nuestro kernel tiene mas throughput por operacion. Si contamos las fmul y las instrucciones predicadas, tiene mucho peor throughput. __expf tiene dos fmul que pueden fusionarse, pero que el compilador ha decidido no hacer. No hay que confundir la latencia con el throughput. Throughput es el numero de resultados que podemos obtener por ciclo por sm por pipeline determinado. Latencia es el tiempo que tarda una operacion. Nvidia nos facilita el throughput de las instrucciones, pero no cuanto tarda una de ellas, ni que ocurre cuando hay dependencias. Solo podemos afirmar que tenemos mas throughput que la expf base. 

Son 2 fmul, 2 fmul opcionales, 1 fstep y una mfu

# Calcular S y downscale a P

# Multiplicar P * V

# Division por reciproco de O. 

# Volcado de O en memoria principal.

# Optimizacion de hiperparametros mediante Optuna.
Para balancear el numero de warps, el numero de ctas que ocupan un sm, cuantos bufferes tenemos en nuestra estrategia multibuffer, cuanto podemos desenrollar el bucle principal sin caer en excesivos fallos de cache de instrucciones, si debemos utilizar accesos cg o wb a las matrices Q y O necesitamos hacer una busqueda de parametros:

| Parámetro                  | Nº de valores | Valores posibles                                        |
| -------------------------- | ------------: | ------------------------------------------------------- |
| `ctas_per_sm`              |             8 | `1, 2, 3, 4, 5, 6, 7, 8`                                |
| `num_warp`                 |             5 | `1, 2, 4, 8, 16`                                        |
| `buffer_num`               |             5 | `2, 3, 4, 5, 6`                                         |
| `uncached`                 |             2 | `0, 1`                                                  |
| `buffer_unroll`            |            16 | `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16` |
| **Total de combinaciones** |      **6400** | \(8 \times 5 \times 5 \times 2 \times 16\)              

A priori, los parametros mencionados podian parecer pocos. Sin embargo, probar 6400 combinaciones es un proceso costoso. Podemos considerar unicamente combinaciones que sean posibles, contando solo el maximo numero de warps por sm, asi como el maximo numero de ctas teniendo en cuenta la sram que ocupa un kernel dado `buffer_num`. 

| Filtro                   | Condición de poda                            | Cómo se calcula                                                                                                                                                                                                                                                                                           | Trials eliminados | Trials restantes |
|--------------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|-----------------:|
| **Threads por bloque**   | `32 * warps_per_block > 1024`                | El máximo posible es `32 * 16 = 512`, así que ninguna combinación viola el límite                                                                                                                                                                                                                         |             **0** |         **6400** |
| **Threads por SM**       | `32 * warps_per_block * ctas_per_sm > 1536`  | Equivale a `warps_per_block * ctas_per_sm > 48`. Hay **7 pares inválidos** `(ctas_per_sm, warps_per_block)`. Cada par fijo todavía puede combinarse con `5` valores de `buffer_num`, `2` de `uncached` y `16` de `buffer_unroll`, es decir `5 * 2 * 16 = 160` trials por par. Por tanto: `7 * 160 = 1120` |          **1120** |         **5280** |
| **Shared memory por SM** | `8 KiB * buffer_num * ctas_per_sm > 100 KiB` | Tras aplicar el filtro anterior, hay **84 combinaciones inválidas** `(ctas_per_sm, warps_per_block, buffer_num)`. Cada una todavía puede combinarse con `2` valores de `uncached` y `16` de `buffer_unroll`, es decir `2 * 16 = 32` trials. Por tanto: `84 * 32 = 2688`                                   |          **2688** |         **2592** |
| **Total**                | —                                            | `1120 + 2688`                                                                                                                                                                                                                                                                                             |          **3808** |         **2592** |

Si tenemos 2592 trials, y suponemos un rango generoso de tiempo de compilado + profiling -30s-, evaluar todas las opciones nos tardaria 2592 trials * 30s/trial = 77760 segundos; 77760 segundos * 1h / 3600s = 21,6h.

21,6h para un rango de parametros tan bajo es un coste inasumible a la par que innecesario. Por ello, debemos hacer una busqueda informada. En el proyecto he decidido realizarla mediante optuna y el sampler TPEMultiSampler, semilla 0, 200 estudios y 100 estudios de busqueda. 


Esta idea se puede extrapolar para ajustar si es necesario warp coarsening e incluso que distribucion de datos le corresponde a cada grafica en una configuracion multigpu. 

# Analisis de los resultados de optuna.

## Longitud de secuencia 8192
## Longitud de secuencia 16k
## Longitud de secuencia 32k
## Longitud de secuencia 64k

# Analisis de los resultados de NCU.

## Longitud de secuencia 8192
## Longitud de secuencia 16k
## Longitud de secuencia 32k
## Longitud de secuencia 64k

# Conclusion.


# Apendice A
Que significa la formula de la atencion

# Apendice B
Que cambia en SM_100

# Apendice C
Branch para recalcular O.

# Apendice D
Aspectos empleados IEE754

