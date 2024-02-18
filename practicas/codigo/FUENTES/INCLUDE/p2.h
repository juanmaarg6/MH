// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
// Problema: APC

#include <functional>
#include <iterator>
#include <set>
#include "p1.h"

using namespace std;

// *************************************************************************
// ************************** Constantes Globales **************************
// *************************************************************************

// Coeficiente del límite superior de generación consecutiva de vecinos en el algoritmo de BL de baja intensidad
const int COEF_MAX_VECINOS_BAJA_INTENSIDAD = 2;

// Parámetro del operador de cruce BLX
const float ALPHA_BLX = 0.3;

// Probabilidad de cruce
const float PROBABILIDAD_CRUCE = 0.7;

// Probabilidad de mutación (por individuo)
const float PROBABILIDAD_MUTACION = 0.1;

// Tamaño de la población (intermedia) en los algoritmos genéticos generacionales
const int TAMANIO_AGG = 50;

// Tamaño de la población (intermedia) en los algoritmos genéticos estacionarios
const int TAMANIO_AGE = 2;

// Tamaño de la población para algoritmos meméticos
const int TAMANIO_AM = 50;

// Frecuencia de aplicación de la búsqueda local en los algoritmos meméticos
const int FRECUENCIA_BL_AM = 10;

// Probabilidad de la búsqueda local en los algoritmos meméticos
const float PROBABILIDAD_BUSQUEDA_LOCAL_AM = 0.1;



// *************************************************************************
// ************************** Estructura de Datos **************************
// *************************************************************************

// Cromosoma
struct Cromosoma {
  vector<double> w;  // Vector de pesos 
  double fitness;     // Valor de la función objetivo para w
};

// Comparación de cromosomas
struct CromosomaComp {
  bool operator()(const Cromosoma& lhs, const Cromosoma& rhs) const {
    return lhs.fitness < rhs.fitness;
  }
};

// Población
typedef multiset<Cromosoma, CromosomaComp> Poblacion;

// Población Intermedia (no evaluada)
typedef vector<Cromosoma> PoblacionIntermedia;



// *************************************************************************
// ******************* Búsqueda Local de Baja Intensidad *******************
// *************************************************************************

/**
 * @fn busquedaLocalBajaIntensidad
 * @brief Calcula el vector de pesos w usando Búsqueda Local de Baja Intensidad
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param c Cromosoma
 * @cond w.size() == entrenamiento[i].num_caracts
 * @return Número de evaluaciones de la función objetivo
 */
int busquedaLocalBajaIntensidad(const vector<Ejemplo>& entrenamiento, Cromosoma& c);



// *************************************************************************
// ************************* Operadores Genéticos **************************
// *************************************************************************

/**
 * @fn inicializarPoblacion
 * @brief Inicializa una población
 * @param p Población a inicializar
 * @param num_cromosomas Número de cromosomas de la población
 * @param tamanio Tamaño del vector de pesos de cada cromosoma
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 */
void inicializarPoblacion(Poblacion& p, int num_cromosomas, int tamanio, const vector<Ejemplo>& entrenamiento);

/**
 * @fn seleccion
 * @brief Operador de selección (Torneo Binario con Reemplazamiento)
 * @param p Población
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @return Cromosoma seleccionado
 */
Cromosoma seleccion(const Poblacion& p);

/**
 * @fn cruceBLX
 * @brief Operador de cruce BLX. Genera dos descendientes por cada dos padres
 * @param c1 Primer cromosoma padre
 * @param c2 Segundo cromosoma padre
 * @cond c1.w.size() == c2.w.size()
 * @return Dos cromosomas descendientes de los dos cromosomas padres
 */
pair<Cromosoma, Cromosoma> cruceBLX(const Cromosoma& c1, const Cromosoma& c2);

/**
 * @fn cruceAritmetico
 * @brief Operador de cruce aritmético. Genera un descendiente por cada dos padres
 * @param c1 Primer cromosoma padre
 * @param c2 Segundo cromosoma padre
 * @cond c1.w.size() == c2.w.size()
 * @return Un cromosoma descendiente de los dos cromosomas padres
 */
Cromosoma cruceAritmetico(const Cromosoma& c1, const Cromosoma& c2);

/**
 * @fn mutacion
 * @brief Operador de mutación. Muta un gen dado de un cromosoma
 * @param c Cromosoma
 * @param comp Componente a mutar del cromosoma
 */
void mutacion(Cromosoma& c, int comp);

/**
 * @fn mutacionesEsperadas
 * @brief Devuelve el número de mutaciones esperadas. Método de redondeo personalizado
 * @param total_genes Número de genes totales
 * @return Número de mutaciones esperadas
 */
int mutacionesEsperadas(int total_genes);



// *************************************************************************
// *************** Algoritmos Genéticos Generacionales (AGGs) **************
// *************************************************************************

/**
 * @fn AGG_BLX
 * @brief AGG usando operador de cruce BLX
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos a calcular
 * @return Número total de generaciones
 */
vector<double> AGG_BLX(const vector<Ejemplo> entrenamiento);

/**
 * @fn AGG_Arit
 * @brief AGG usando operador de cruce aritmético
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos a calcular
 * @return Número total de generaciones
 */
vector<double>  AGG_Arit(const vector<Ejemplo> entrenamiento);



// *************************************************************************
// *************** Algoritmos Genéticos Estacionarios (AGEs) ***************
// *************************************************************************

/**
 * @fn AGE_BLX
 * @brief AGE usando operador de cruce BLX
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos a calcular
 * @return Número total de generaciones
 */
vector<double> AGE_BLX(const vector<Ejemplo> entrenamiento);


/**
 * @fn AGE_Arit
 * @brief AGE usando operador de cruce artimético
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos a calcular
 * @return Número total de generaciones
 */
vector<double> AGE_Arit(const vector<Ejemplo> entrenamiento);



// *************************************************************************
// ********************** Algoritmos Meméticos (AMMs) **********************
// *************************************************************************

/**
 * @fn AM_All
 * @brief Cada 10 generaciones, aplica la BL sobre todos los cromosomas de la población. AM-(10,1.0)
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos a calcular
 * @return Número total de generaciones
 */
vector<double> AM_All(const vector<Ejemplo> entrenamiento);


/**
 * @fn AM_Rand
 * @brief Cada 10 generaciones, aplica la BL sobre un subconjunto de cromosomas de la población 
 *        seleccionado aleatoriamente con probabilidad 0.1 para cada cromosoma. AM-(10,0.1)
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos a calcular
 * @return Número total de generaciones
 */
vector<double> AM_Rand(const vector<Ejemplo> entrenamiento);

/**
 * @fn AM_Best
 * @brief Cada 10 generaciones, aplica la BL sobre los 0.1*N mejores cromosomas de la población 
 *        actual (N es el tamaño de dicha población). AM-(10,0.1mej)
 * @param entrenamiento Conjunto de ejemplos de entrenamiento
 * @param w Vector de pesos a calcular
 * @return Número total de generaciones
 */
vector<double> AM_Best(const vector<Ejemplo> entrenamiento);



// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

/**
 * @fn resultadosAGG_BLX
 * @brief Calcula y muestra los resultados del algoritmo AGG_BLX
 */
void resultadosAGG_BLX();

/**
 * @fn resultadosAGG_Arit
 * @brief Calcula y muestra los resultados del algoritmo AGG_Arit
 */
void resultadosAGG_Arit();

/**
 * @fn resultadosAGE_BLX
 * @brief Calcula y muestra los resultados del algoritmo AGE_BLX
 */
void resultadosAGE_BLX();

/**
 * @fn resultadosAGE_Arit
 * @brief Calcula y muestra los resultados del algoritmo AGE_Arit
 */
void resultadosAGE_Arit();

/**
 * @fn resultadosAM_All
 * @brief Calcula y muestra los resultados del algoritmo AM_All
 */
void resultadosAM_All();

/**
 * @fn resultadosAM_Rand
 * @brief Calcula y muestra los resultados del algoritmo AM_Rand
 */
void resultadosAM_Rand();

/**
 * @fn resultadosAM_Best
 * @brief Calcula y muestra los resultados del algoritmo AM_Best
 */
void resultadosAM_Best();