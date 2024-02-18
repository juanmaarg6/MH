// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica Alternativa al Examen: Metaheurística Leaders and Followers
// Problema: APC

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

using namespace std;

// *************************************************************************
// ************************** Constantes Globales **************************
// *************************************************************************

// Número de archivos diferentes (cada uno es un dataset diferente)
const int NUM_ARCHIVOS = 3;

// Número de particiones de cada archivo
const int NUM_PARTICIONES = 5;

// Ponderación de la importancia entre el acierto y la reducción de la solución encontrada
const float ALPHA = 0.8;



// *************************************************************************
// ************************** Estructura de Datos **************************
// *************************************************************************

/**
 * @struct Ejemplo
 * @brief Representa un ejemplo de nuestro conjunto de datos
 */
struct Ejemplo {
    vector<double> val_caracts;   // Valores reales de las características
    string categoria;             // Categoría o clase a la que pertenece
    int num_caracts;              // Número de características
};



// *************************************************************************
// ******************** Lectura y Normalización de Datos *******************
// *************************************************************************

/**
 * @fn leerFicheroARFF
 * @brief Lee un archivo ARFF
 * @param nombre_archivo Nombre del archivo a leer
 * @return Vector de ejemplos leídos del archivo ARFF
 */
vector<Ejemplo> leerFicheroARFF(string nombre_archivo);

/**
 * @fn normalizarValores
 * @brief Normaliza los datos calculando el mínimo y el máximo entre todos los datasets
 * @param datos Conjunto de datos a normalizar
 */
void normalizarValores(vector<vector<Ejemplo>>& datos);



// *************************************************************************
// ********************* Inicializar Vector de Pesos ***********************
// *************************************************************************

/**
 * @fn inicializarVectorPesos
 * @brief Inicializa las componentes del vector de pesos a 0.0
 * @param w Vector de pesos a inicializar
 */
void inicializarVectorPesos(vector<double>& w);



// *************************************************************************
// ************************** Funciones Distancia **************************
// *************************************************************************

/**
 * @fn distanciaEuclidea
 * @brief Distancia euclidea entre dos ejemplos
 * @param e1 Un ejemplo del conjunto de datos
 * @param e2 Otro ejemplo del conjunto de datos
 * @cond e1.num_caracts == e2.num_caracts
 * @return Distancia euclídea entre e1 y e2
 */
double distanciaEuclidea(const Ejemplo& e1, const Ejemplo& e2);

/**
 * @fn distanciaEuclideaPesos
 * @brief Distancia euclidea entre dos ejemplos considerando los pesos
 * @param e1 Un ejemplo del conjunto de datos
 * @param e2 Otro ejemplo del conjunto de datos
 * @param w Vector de pesos
 * @cond e1.num_caracts == e2.num_caracts
 * @return Distancia euclídea entre e1 y e2
 */
double distanciaEuclideaPesos(const Ejemplo& e1, const Ejemplo& e2, const vector<double>& w);



// *************************************************************************
// *************************** Clasificador 1-NN ***************************
// *************************************************************************

/**
 * @fn clasificador1NNPesos
 * @brief Calcula la clase de un ejemplo en función de la clase de su vecino más cercano teniendo en cuenta los pesos
 * @param e Ejemplo a clasificar
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos
 * @cond e.num_caracts == entrenamiento[i].num_caracts
 * @return Clase del ejemplo e
 */
string clasificador1NNPesos(const Ejemplo& e, const vector<Ejemplo>& entrenamiento, const vector<double>& w);



// *************************************************************************
// ************************* Función de Evaluación *************************
// *************************************************************************

/**
 * @fn tasaClasificacion
 * @brief Calcula la tasa de clasificación
 * @param test Conjunto de ejemplos de test
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos
 * @return Valor de la tasa de clasificación
 */
double tasaClasificacion(const vector<Ejemplo>& test, const vector<Ejemplo>& entrenamiento, const vector<double> &w);

/**
 * @fn tasaClasificacionLeaveOneOut
 * @brief Calcula la tasa de clasificación usando la técnica Leave-One-Out
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos
 * @return Valor de la tasa de clasificación
 */
double tasaClasificacionLeaveOneOut(const vector<Ejemplo>& entrenamiento,const vector<double>& w);

/**
 * @fn tasaReduccion
 * @brief Calcula la tasa de reduccion
 * @param w Vector de pesos
 * @return Valor de la tasa de reducción
 */
double tasaReduccion(const vector<double>& w);

/**
 * @fn fitness
 * @brief Calcula la función objetivo
 * @param tasa_clas Tasa de clasificación
 * @param tasa_red Tasa de reducción
 * @return Valor de la función objetivo
 */
double fitness(double tasa_clas, double tasa_red);

