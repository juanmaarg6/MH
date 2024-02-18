// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
// Problema: APC

#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <cmath>
#include <algorithm>
#include "util.h"
#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

// *************************************************************************
// ************************** Constantes Globales **************************
// *************************************************************************

// Desviación típica para distribución normal
const float SIGMA = sqrt(0.3);

// Máximo número de iteraciones para el algoritmo de BL
const int MAX_ITER = 15000;

// Coeficiente del límite superior de generación consecutiva de vecinos en el algoritmo de BL
const int COEF_MAX_VECINOS = 20;



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
// ***************************** Greedy RELIEF *****************************
// *************************************************************************

/**
 * @fn amigoMasCercano
 * @brief Calcula el vecino más cercano de la misma clase (amigo) a un ejemplo dado
 * @param e Ejemplo dado
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @cond e.num_caracts == entrenamiento[i].num_caracts
 * @return Vecino más cercano de la misma clase (amigo) con respecto al ejemplo e
 */
Ejemplo amigoMasCercano(const Ejemplo& e,const vector<Ejemplo>& entrenamiento);

/**
 * @fn enemigoMasCercano
 * @brief Calcula el vecino más cercano de diferente clase (enemigo) a un ejemplo dado
 * @param e Ejemplo dado
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @cond e.num_caracts == entrenamiento[i].num_caracts
 * @return Vecino más cercano de diferente clase (enemigo) con respecto al ejemplo e
 */
Ejemplo enemigoMasCercano(const Ejemplo& e,const vector<Ejemplo>& entrenamiento);

/**
 * @fn greedy
 * @brief Calcula el vector de pesos w usando Greedy RELIEF
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @cond w.size() == entrenamiento[i].num_caracts
 * @return Vector de pesos
 */
vector<double> greedy(const vector<Ejemplo>& entrenamiento);



// *************************************************************************
// ************************** Búsqueda Local (BL) **************************
// *************************************************************************

/**
 * @fn busquedaLocal
 * @brief Calcula el vector de pesos w usando Búsqueda Local con el esquema del Primer Mejor
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @cond w.size() == entrenamiento[i].num_caracts
 * @return Vector de pesos
 */
vector<double> busquedaLocal(const vector<Ejemplo>& entrenamiento);



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



// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

/**
 * @fn resultados1NN
 * @brief Calcula y muestra los resultados del algoritmo 1-NN
 */
void resultados1NN();

/**
 * @fn resultadosGreedy
 * @brief Calcula y muestra los resultados del algoritmo Greedy RELIEF
 */
void resultadosGreedy();

/**
 * @fn resultadosBusquedaLocal
 * @brief Calcula y muestra los resultados del algoritmo Búsqueda Local
 */
void resultadosBusquedaLocal();