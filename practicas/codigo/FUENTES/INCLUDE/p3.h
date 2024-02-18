// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
// Problema: APC

#include "p2.h"

using namespace std;

// *************************************************************************
// ************************** Constantes Globales **************************
// *************************************************************************

// ********************** Búsqueda Local K Componentes *********************

// Máximo número de iteraciones para el algoritmo de BL
const int MAX_ITER_BL_K_COMP = 1000;



// ****************** Búsqueda Multiarranque Básica (BMB) ******************

// Número de iteraciones para el algoritmo BMB
const int ITER_BMB = 15;



// *********************** Enfriamiento Simulado (ES) **********************

// Parámetros para calcular la temperatura inicial
const float PHI = 0.2;
const float MU = 0.3;

// Coeficiente del límite superior de generación consecutiva de vecinos en el algoritmo de ES
const int COEF_MAX_VECINOS_ES = 10;

// Coeficiente del límite superior de generaciones con éxito por vecino
const float COEF_MAX_EXITO = 0.1;

// Máximo número de iteraciones para el algoritmo de ES
const int MAX_ITER_ES = 15000;

// Máximo número de iteraciones para el algoritmo de ES aplicado a una solución
const int MAX_ITER_ES_SOL = 1000;



// ********************* Búsqueda Local Iterativa (ILS) ********************

// Número de iteraciones para el algoritmo ILS
const int ITER_ILS = 15;

// Percentage of traits to mutate
const float FACTOR_MUTACION_ILS = 0.1;



// ****************** Búsqueda de Vecindario Variable (VNS) ****************

// Número de iteraciones para el algoritmo VNS
const int ITER_VNS = 15;

// Número máximo de componentes del vector de pesos a modificar
const int K_MAX = 3;



// *************************************************************************
// ************************** Estructura de Datos **************************
// *************************************************************************

// Solución
struct Solucion {
  vector<double> w;   // Vector de pesos 
  double fitness;     // Valor de la función objetivo para w
};



// *************************************************************************
// ************************** Operadores Comunes ***************************
// *************************************************************************

/**
 * @fn inicializarSolucion
 * @brief Inicializa una solución
 * @param sol Solución a inicializar
 * @param tamanio Tamaño del vector de pesos de cada solución
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 */
void inicializarSolucion(Solucion& sol, int tamanio, const vector<Ejemplo>& entrenamiento);

/**
 * @fn mutacionFuerte
 * @brief Operador de mutación fuerte. Cambia el peso del un 10% de las características
 *        de un vector de pesos escogidas aleatoriamente, cambiándolas por valores
 *        aleatorios entre 0 y 1 según una distribución uniforme
 * @param w Vector de pesos
 */
void mutacionFuerte(vector<double>& w);



// *************************************************************************
// ********************** Búsqueda Local K Componentes *********************
// *************************************************************************

/**
 * @fn busquedaLocalKcomp
 * @brief Calcula una solución usando Búsqueda Local con el esquema del Primer Mejor
 *        Las diferencias de esta BL con la BL de la Práctica 1 son las siguientes:
 *           - Ahora el máximo número de iteraciones para el algoritmo en vez de ser 15000 es 1000 
 *           - Podemos modificar k componentes en vez de solo una
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param sol Solución a calcular
 * @param k Número de componentes del vector de pesos a modificar
 * @cond sol.w.size() == entrenamiento[i].num_caracts
 */
void busquedaLocalKcomp(const vector<Ejemplo>& entrenamiento, Solucion& sol, int k);



// *************************************************************************
// ******************* Búsqueda Multiarranque Básica (BMB) *****************
// *************************************************************************

/**
 * @fn BMB
 * @brief Algoritmo de Búsqueda Multiarranque Básica
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> BMB(const vector<Ejemplo>& entrenamiento);



// *************************************************************************
// *********************** Enfriamiento Simulado (ES) **********************
// *************************************************************************

/**
 * @fn ES
 * @brief Algoritmo de Enfriamiento Simulado
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> ES(const vector<Ejemplo>& entrenamiento);

/**
 * @fn ES_Solucion
 * @brief Algoritmo de Enfriamiento Simulado aplicado a una solución (se usará en ILS_ES)
 * @param entrenamiento Conjunto de ejemplos de 
 * @param sol Solución
 */
void ES_Solucion(const vector<Ejemplo>& entrenamiento, Solucion& sol);



// *************************************************************************
// ********************* Búsqueda Local Iterativa (ILS) ********************
// *************************************************************************

/**
 * @fn ILS
 * @brief Algoritmo de Búsqueda Local Iterativa usando el algoritmo de BL 
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> ILS(const vector<Ejemplo>& entrenamiento);

/**
 * @fn ILS_ES
 * @brief Algoritmo de Búsqueda Local Iterativa usando el algoritmo de ES 
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> ILS_ES(const vector<Ejemplo>& entrenamiento);



// *************************************************************************
// ****************** Búsqueda de Vecindario Variable (VNS) ****************
// *************************************************************************

/**
 * @fn VNS
 * @brief Algoritmo de Búsqueda de Vecindario Variable
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> VNS(const vector<Ejemplo>& entrenamiento);



// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

/**
 * @fn resultadosBMB
 * @brief Calcula y muestra los resultados del algoritmo BMB
 */
void resultadosBMB();

/**
 * @fn resultadosES
 * @brief Calcula y muestra los resultados del algoritmo ES
 */
void resultadosES();

/**
 * @fn resultadosILS
 * @brief Calcula y muestra los resultados del algoritmo ILS
 */
void resultadosILS();

/**
 * @fn resultadosILS_ES
 * @brief Calcula y muestra los resultados del algoritmo ILS_ES
 */
void resultadosILS_ES();

/**
 * @fn resultadosVNS
 * @brief Calcula y muestra los resultados del algoritmo VNS
 */
void resultadosVNS();