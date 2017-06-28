# Reto MNIST
## Descripcion
El [MNIST](http://yann.lecun.com/exdb/mnist/) es un conjunto de imágenes de digitos del 0 al 9 escritos a mano.

![alt text][s1]

Este dataset tiene mas de 60,000 imágenes separadas en 10 clases. El reto es construir un clasificador de imágenes que sea capaz de reconocer los digitos.

### Variables
Cada imagen como tal puede ser representada por una matriz de dimensiones `height x width` dado que estan en escala de grises.


### Objetivo
1. Crear un algoritmo que tome una imagen de entrada, ya sea como vector o matriz, y retorne el clase (`class_id`) a la que pertenece esa imagen.
1. Entrenar este algoritmo utilizando los datos de la carpeta `data/training-set`.
1. Medir el performance/score del algoritmo utilizando los datos de la carpeta `data/test-set`. El performance debe ser medido como
```python
score = n_aciertos / n_imagenes * 100
```
donde `n_aciertos` es el numero de imagenes clasificadas de forma correcta y `n_imagenes` es el numero total de imagenes en el `test-set`.

### Solución

### Requerimientos
Para descargar y visualizar los datos necesitas Python 2 o 3. Las dependencias las puedes encontrar en el archivo `requirements.txt`, el cual incluye
* pillow
* numpy
* pandas
* jupyter

Para crear el modelo solo necesitas scikit-learn. Puedes instalarlas fácilmente utilizando el commando

```bash
pip install -r requirements.txt
```
Dependiendo de tu entorno puede que necesites instalar paquetes del sistema adicionales, si tienes problemas revisa la documentación de estas librerías.

### Descarga y Preprocesamiento
Para descargar los datos ejecuta el comando
```bash
dataget get mnist
```
Esto descarga los archivos en la carpeta `.dataget/data`, los divide en los conjuntos `training-set` y `test-set`, convierte las imagenes en `jpg` de dimensiones `32x32`.


##### Procedimiento
Para entrenar el modelo y a la vez validarlo ejecutar el siguiente comando de consola:

```
python model.py
```

El entrenamiento puede demorar alrededor de 30 minutos.

##### Método
Se usó una máquina de soporte vectorial con kernel radial (RBF) con parámetros:
* C = 5
* gamma = 0.05

La entrada al modelo son las imágenes (en forma de vector aplanadas) pero transformando la magnitud de los pixeles entre 0 y 1. Los parámetros fueron consultados de otras soluciones en la Web.

##### Resultados
Resultado: **0.9837**



# Starter Code Python
Para iniciar con este reto puedes correr el codigo de Python en Jupyter del archivo `python-sample.ipynb`. Este código que ayudará a cargar y visualizar algunas imágenes. Las dependencias son las mismas que se instalaron durante la descarga de los datos, ver [Requerimientos](#requerimientos).

Para iniciar el código solo hay que prender Jupyter en esta carpeta

```bash
jupyter notebook .
```
y abrir el archivo `python-sample.ipynb`.



[s1]: http://rodrigob.github.io/are_we_there_yet/build/images/mnist.png?1363085077 "S"
