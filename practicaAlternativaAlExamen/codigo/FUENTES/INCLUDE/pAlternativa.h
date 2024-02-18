// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica Alternativa al Examen: Metaheurística Leaders and Followers
// Problema: APC

#include <iostream>
#include <iomanip>
#include <fstream>
#include <set>
#include "util.h"
#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

// *************************************************************************
// ************************** Constantes Globales **************************
// *************************************************************************

// Máximo número de iteraciones para el algoritmo de Leaders and Followers
const int MAX_ITER_LEADERS_AND_FOLLOWERS = 20;

// Coeficiente del límite superior de generación consecutiva de vecinos en el algoritmo de BL de baja intensidad
const int COEF_MAX_VECINOS_BAJA_INTENSIDAD = 2;

// Desviación típica para distribución normal
const float SIGMA = sqrt(0.3);



// *************************************************************************
// ************************** Estructura de Datos **************************
// *************************************************************************

// Leader o Follower
struct LF {
  vector<double> w;  // Vector de pesos 
  double fitness;     // Valor de la función objetivo para w
};

// Comparación de Leaders/Followers
struct LFComp {
  bool operator()(const LF& lhs, const LF& rhs) const {
    return lhs.fitness < rhs.fitness;
  }
};

// Población
typedef multiset<LF, LFComp> PoblacionLF;



// *************************************************************************
// ******************* Búsqueda Local de Baja Intensidad *******************
// *************************************************************************

/**
 * @fn busquedaLocalBajaIntensidad
 * @brief Calcula el vector de pesos usando Búsqueda Local de Baja Intensidad
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param individuo Leader o Follower
 * @cond individuo.w.size() == entrenamiento[i].num_caracts
 * @return Número de evaluaciones de la función objetivo
 */
int busquedaLocalBajaIntensidad(const vector<Ejemplo>& entrenamiento, LF& individuo);



// *************************************************************************
// ************************* Funciones Auxiliares **************************
// *************************************************************************

/**
 * @fn inicializarPoblacionLF
 * @brief Inicializa una población de Leaders o de Followers
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Población de Leaders o de Followers inicializada
 */
PoblacionLF inicializarPoblacionLF(const vector<Ejemplo>& entrenamiento);

/**
 * @fn createTrial
 * @brief Calcula un nuevo individuo en base a un leader y un follower
 * @param leader Leader
 * @param follower Follower
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Nuevo individuo, llamado trial
 */
LF createTrial(const LF& leader, const LF& follower, const vector<Ejemplo>& entrenamiento);

/**
 * @fn createTrialModificado
 * @brief Calcula un nuevo individuo en base a un leader y un follower (Modificación de la función createTrial)
 * @param leader Leader
 * @param follower Follower
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param mutacion Tamaño de mutación de la población
 * @return Nuevo individuo, llamado trial
 */
LF createTrialModificado(const LF& leader, const LF& follower, const vector<Ejemplo>& entrenamiento, double mutacion);


/**
 * @fn calcularMediana
 * @brief Calcula la mediana del fitness de una población
 * @param p Población
 * @return Mediana del fitness de una población
 */
double calcularMediana(const PoblacionLF& p);

/**
 * @fn calcularMediana
 * @brief Combina ambas poblaciones manteniendo el mejor leader y realizando un torneo
 *        binario sin reemplazamiento, con selección aleatoria, para los n-1 individuos restantes
 * @param leaders Población de Leaders
 * @param followers Población de Followers
 * @return Combinación de ambas poblaciones
 */
PoblacionLF mergePopulations(const PoblacionLF& leaders, const PoblacionLF& followers);



// *************************************************************************
// ***************** Algoritmos Leaders and Followers (LFs) ****************
// *************************************************************************

/**
 * @fn LeadersAndFollowers
 * @brief Algoritmo Leaders and Followers
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> LeadersAndFollowers(const vector<Ejemplo>& entrenamiento);

/**
 * @fn LeadersAndFollowersBL
 * @brief Algoritmo Leaders and Followers + Búsqueda Local de Baja Intensidad (Hibridación)
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> LeadersAndFollowersBL(const vector<Ejemplo>& entrenamiento);

/**
 * @fn LeadersAndFollowersModificado
 * @brief Algoritmo Leaders and Followers + Modificado
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Vector de pesos
 */
vector<double> LeadersAndFollowersModificado(const vector<Ejemplo>& entrenamiento);



// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

/**
 * @fn resultadosLeadersAndFollowers
 * @brief Calcula y muestra los resultados del algoritmo LeadersAndFollowers
 */
void resultadosLeadersAndFollowers();

/**
 * @fn resultadosLeadersAndFollowersBL
 * @brief Calcula y muestra los resultados del algoritmo LeadersAndFollowersBL
 */
void resultadosLeadersAndFollowersBL();

/**
 * @fn resultadosLeadersAndFollowersModificado
 * @brief Calcula y muestra los resultados del algoritmo LeadersAndFollowersModificado
 */
void resultadosLeadersAndFollowersModificado();