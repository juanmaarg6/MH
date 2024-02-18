// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
// Problema: APC

#include "p3.h"

using namespace std;

// *************************************************************************
// ************************** Operadores Comunes ***************************
// *************************************************************************

void inicializarSolucion(Solucion& sol, int tamanio, const vector<Ejemplo>& entrenamiento) {

  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);

  sol.w.resize(tamanio);

  for(int i = 0; i < tamanio; ++i)
    sol.w[i] = Random::get(distribucion_uniform_real);

  sol.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol.w), tasaReduccion(sol.w));
}

void mutacionFuerte(vector<double>& w) {
  
  int num_mutaciones = FACTOR_MUTACION_ILS * w.size();
  num_mutaciones = min(num_mutaciones, 2);  // Habrá al menos 2 componentes a mutar

  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);

  vector<int> indices;

  for(int i = 0; i < w.size(); ++i)
    indices.push_back(i);

  Random::shuffle(indices);

  for(int i = 0; i < num_mutaciones; ++i)
    w[indices[i]] = Random::get(distribucion_uniform_real);
}



// *************************************************************************
// ********************** Búsqueda Local K Componentes *********************
// *************************************************************************

void busquedaLocalKcomp(const vector<Ejemplo>& entrenamiento, Solucion& sol, int k) {

  vector<int> indices;    
  double mejor_fitness;

  int num_iteraciones = 0;
  int num_iteraciones_sin_mejora = 0;

  bool hay_mejora = false;

  normal_distribution<double> distribucion_normal(0.0, SIGMA);


  // Inicializamos el vector de índices
  for(int i = 0; i < sol.w.size(); ++i)
    indices.push_back(i);

  Random::shuffle(indices);

  mejor_fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol.w), tasaReduccion(sol.w));

  // Búsqueda en el vecindario del primero mejor
  while( (num_iteraciones_sin_mejora < sol.w.size() * COEF_MAX_VECINOS) && (num_iteraciones < MAX_ITER_BL_K_COMP) ) {

    Solucion sol_mutada = sol;

    // Seleccionamos k componentes de w para mutar
    for(int i = 0; i < k; ++i) {
      int comp_mutada = indices[num_iteraciones % sol.w.size()];

      sol_mutada.w[comp_mutada] += Random::get(distribucion_normal);

      // Truncamos el peso de la componente mutada si fuera necesario
      if(sol_mutada.w[comp_mutada] > 1) 
        sol_mutada.w[comp_mutada] = 1;
      else if(sol_mutada.w[comp_mutada] < 0) 
        sol_mutada.w[comp_mutada] = 0;

      num_iteraciones++;
    }

    sol_mutada.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol_mutada.w), tasaReduccion(sol_mutada.w));

    if(sol_mutada.fitness > mejor_fitness) {
      num_iteraciones_sin_mejora = 0;
      sol = sol_mutada;
      mejor_fitness = sol_mutada.fitness;
      hay_mejora = true;
    }
    else
      num_iteraciones_sin_mejora++;

    // Actualizamos el vector de índices si ha habido mejora o ya se ha recorrido entero sin encontrar mejora
    if(num_iteraciones >= sol.w.size() == 0 || hay_mejora) {
      Random::shuffle(indices);
      hay_mejora = false;
    }
  }
}



// *************************************************************************
// ******************* Búsqueda Multiarranque Básica (BMB) *****************
// *************************************************************************

vector<double> BMB(const vector<Ejemplo>& entrenamiento) {

  vector<double> w(entrenamiento[0].num_caracts);

  Solucion mejor_solucion;

  for(int i = 0; i < ITER_BMB; ++i) {

    // Generar solución aleatoria
    Solucion sol;
    inicializarSolucion(sol, w.size(), entrenamiento);

    // Aplicar Búsqueda Local a la solución generada
    busquedaLocalKcomp(entrenamiento, sol, 1);

    // Criterio de aceptación
    if(sol.fitness > mejor_solucion.fitness)
      mejor_solucion = sol;
  }

  w = mejor_solucion.w;

  return w;
}



// *************************************************************************
// *********************** Enfriamiento Simulado (ES) **********************
// *************************************************************************

vector<double> ES(const vector<Ejemplo>& entrenamiento) {

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;

  uniform_int_distribution<int> distribucion_uniform_int(0, w.size() - 1);
  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);
  normal_distribution<double> distribucion_normal(0.0, SIGMA);

  // Inicializar solución y temperatura
  Solucion sol;

  inicializarSolucion(sol, w.size(), entrenamiento);
  num_iteraciones++;

  Solucion mejor_solucion = sol;

  const float TEMP_INICIAL = (MU * mejor_solucion.fitness) / (- 1.0 * log(PHI));
  float temp = TEMP_INICIAL;

  float temp_final = 1e-4;

  while(temp_final >= TEMP_INICIAL)
    temp_final /= 100.0;

  const int MAX_VECINOS = COEF_MAX_VECINOS_ES * w.size();
  const int MAX_EXITOS = COEF_MAX_EXITO * MAX_VECINOS;
  const int M = MAX_ITER_ES / MAX_VECINOS;
  const float BETA = (float) (TEMP_INICIAL - temp_final) / (M * TEMP_INICIAL * temp_final);

  // Bucle Exterior
  int num_exitos = 1;
  int num_vecinos;
  
  while( (num_iteraciones < MAX_ITER_ES) && (num_exitos != 0) ) {

    num_vecinos = 0;
    num_exitos = 0;

    // Bucle Interior (Enfriamiento)
    while( (num_iteraciones < MAX_ITER_ES) && (num_vecinos < MAX_VECINOS) && (num_exitos < MAX_EXITOS) ) {
      
      // Mutar una componente random
      int comp = Random::get(distribucion_uniform_int);

      Solucion sol_mutada = sol;

      sol_mutada.w[comp] += Random::get(distribucion_normal);

      // Truncamos el peso de la componente mutada si fuera necesario
      if(sol_mutada.w[comp] > 1) 
        sol_mutada.w[comp] = 1;
      else if(sol_mutada.w[comp] < 0) 
        sol_mutada.w[comp] = 0;

      sol_mutada.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol_mutada.w), tasaReduccion(sol_mutada.w));

      num_iteraciones++;
      num_vecinos++;

      // Criterio de aceptación
      float dif_fitness = sol.fitness - sol_mutada.fitness;  

      // Evitar siempre aceptar cada vecino si dif_fitness es 0
      if(dif_fitness == 0)
        dif_fitness = 0.001;

      if( (dif_fitness < 0) || Random::get(distribucion_uniform_real) <= exp(-1.0 * dif_fitness / temp)) {
        num_exitos++;
        sol = sol_mutada;
        if(sol.fitness > mejor_solucion.fitness)
          mejor_solucion = sol;
      }
    }

    // Enfriamiento (esquema de Cauchy modificado)
    temp = temp / (1.0 + BETA * temp);
  }

  w = mejor_solucion.w;

  return w;
}

void ES_Solucion(const vector<Ejemplo>& entrenamiento, Solucion& sol) {

  int num_iteraciones = 0;

  uniform_int_distribution<int> distribucion_uniform_int(0, sol.w.size() - 1);
  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);
  normal_distribution<double> distribucion_normal(0.0, SIGMA);

  // Inicializar solución y temperatura
  Solucion mejor_solucion = sol;
  num_iteraciones++;

  const float TEMP_INICIAL = (MU * mejor_solucion.fitness) / (- 1.0 * log(PHI));
  float temp = TEMP_INICIAL;

  float temp_final = 1e-4;

  while(temp_final >= TEMP_INICIAL)
    temp_final /= 100.0;

  const int MAX_VECINOS = COEF_MAX_VECINOS_ES * mejor_solucion.w.size();
  const int MAX_EXITOS = COEF_MAX_EXITO * MAX_VECINOS;
  const int M = MAX_ITER_ES_SOL / MAX_VECINOS;
  const float BETA = (float) (TEMP_INICIAL - temp_final) / (M * TEMP_INICIAL * temp_final);

  // Bucle Exterior
  int num_exitos = 1;
  int num_vecinos;

  while( (num_iteraciones < MAX_ITER_ES_SOL) && (num_exitos != 0) ) {

    num_vecinos = 0;
    num_exitos = 0;

    // Bucle Interior (Enfriamiento)
    while( (num_iteraciones < MAX_ITER_ES_SOL) && (num_vecinos < MAX_VECINOS) && (num_exitos < MAX_EXITOS) ) {
      
      // Mutar una componente random
      int comp = Random::get(distribucion_uniform_int);

      Solucion sol_mutada = sol;

      sol_mutada.w[comp] += Random::get(distribucion_normal);

      // Truncamos el peso de la componente mutada si fuera necesario
      if(sol_mutada.w[comp] > 1) 
        sol_mutada.w[comp] = 1;
      else if(sol_mutada.w[comp] < 0) 
        sol_mutada.w[comp] = 0;

      sol_mutada.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol_mutada.w), tasaReduccion(sol_mutada.w));

      num_iteraciones++;
      num_vecinos++;

      // Criterio de aceptación
      float dif_fitness = sol.fitness - sol_mutada.fitness;  

      // Evitar siempre aceptar cada vecino si dif_fitness es 0
      if(dif_fitness == 0)
        dif_fitness = 0.001;

      if( (dif_fitness < 0) || Random::get(distribucion_uniform_real) <= exp(-1.0 * dif_fitness / temp)) {
        num_exitos++;
        sol = sol_mutada;
        if(sol.fitness > mejor_solucion.fitness)
          mejor_solucion = sol;
      }
    }

    // Enfriamiento (esquema de Cauchy modificado)
    temp = temp / (1.0 + BETA * temp);
  }
  
  sol = mejor_solucion;
}



// *************************************************************************
// ********************* Búsqueda Local Iterativa (ILS) ********************
// *************************************************************************

vector<double> ILS(const vector<Ejemplo>& entrenamiento) {

  vector<double> w(entrenamiento[0].num_caracts);

  uniform_int_distribution<int> distribucion_uniform_int(0, w.size() - 1);
  
  Solucion sol_inicial;

  // Inicializar solución inicial
  inicializarSolucion(sol_inicial, w.size(), entrenamiento);

  // Aplicar Búsqueda Local a la solución inicial
  busquedaLocalKcomp(entrenamiento, sol_inicial, 1);

  Solucion mejor_solucion = sol_inicial;

  for(int i = 1; i < ITER_ILS; ++i) {

    // Mutar algunas componentes
    Solucion sol_mutada = mejor_solucion;

    mutacionFuerte(sol_mutada.w);

    // Reiterar la Búsqueda Local
    sol_mutada.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol_mutada.w), tasaReduccion(sol_mutada.w));
    busquedaLocalKcomp(entrenamiento, sol_mutada, 1);

    // Criterio de aceptación
    if(sol_mutada.fitness > mejor_solucion.fitness)
      mejor_solucion = sol_mutada;
  }

  w = mejor_solucion.w;

  return w;
}

vector<double> ILS_ES(const vector<Ejemplo>& entrenamiento) {

  vector<double> w(entrenamiento[0].num_caracts);

  uniform_int_distribution<int> distribucion_uniform_int(0, w.size() - 1);
  
  Solucion sol_inicial;

  // Inicializar solución inicial
  inicializarSolucion(sol_inicial, w.size(), entrenamiento);

  // Aplicar ES a la solución inicial
  ES_Solucion(entrenamiento, sol_inicial);

  Solucion mejor_solucion = sol_inicial;

  for(int i = 1; i < ITER_ILS; ++i) {

    // Mutar algunas componentes
    Solucion sol_mutada = mejor_solucion;

    mutacionFuerte(sol_mutada.w);

    // Reiterar el ES
    sol_mutada.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol_mutada.w), tasaReduccion(sol_mutada.w));
    ES_Solucion(entrenamiento, sol_mutada);

    // Criterio de aceptación
    if(sol_mutada.fitness > mejor_solucion.fitness)
      mejor_solucion = sol_mutada;
  }

  w = mejor_solucion.w;

  return w;
}



// *************************************************************************
// ****************** Búsqueda de Vecindario Variable (VNS) ****************
// *************************************************************************

vector<double> VNS(const vector<Ejemplo>& entrenamiento) {

  vector<double> w(entrenamiento[0].num_caracts);

  uniform_int_distribution<int> distribucion_uniform_int(0, w.size() - 1);
  
  Solucion sol_inicial;

  int k = 1;

  // Inicializar solución inicial
  inicializarSolucion(sol_inicial, w.size(), entrenamiento);

  // Aplicar Búsqueda Local a la solución inicial
  busquedaLocalKcomp(entrenamiento, sol_inicial, k);

  Solucion mejor_solucion = sol_inicial;

  for(int i = 1; i < ITER_VNS; ++i) {

    // Mutar algunas componentes
    Solucion sol_mutada = mejor_solucion;

    mutacionFuerte(sol_mutada.w);

    // Reiterar la Búsqueda Local
    sol_mutada.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, sol_mutada.w), tasaReduccion(sol_mutada.w));
    busquedaLocalKcomp(entrenamiento, sol_mutada, k);

    // Criterio de aceptación
    if(sol_mutada.fitness > mejor_solucion.fitness) {
      mejor_solucion = sol_mutada;
      k = 1;
    }
    else
      k = (k % K_MAX) + 1;
  }

  w = mejor_solucion.w;

  return w;
}



// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

void resultadosBMB() {

  string nombre_archivo;

  for(int l = 0; l < NUM_ARCHIVOS; ++l) {

    switch(l) {
      case 0:
        nombre_archivo = "diabetes";
      break;

      case 1:
        nombre_archivo = "ozone-320";
      break;

      case 2:
        nombre_archivo = "spectf-heart";
      break;
    }

    // Leer y normalizar dataset
    vector< vector<Ejemplo> > dataset;
    vector<Ejemplo> particion_1 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_1.arff");
    vector<Ejemplo> particion_2 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_2.arff");
    vector<Ejemplo> particion_3 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_3.arff");
    vector<Ejemplo> particion_4 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_4.arff");
    vector<Ejemplo> particion_5 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_5.arff");

    dataset.push_back(particion_1);
    dataset.push_back(particion_2);
    dataset.push_back(particion_3);
    dataset.push_back(particion_4);
    dataset.push_back(particion_5);

    normalizarValores(dataset);

    cout << endl << endl;
    cout << "************************************ " << nombre_archivo << " (BMB) ************************************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio 
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo Búsqueda Local en las diferentes particiones
    for(int i = 0; i < NUM_PARTICIONES; ++i) {
      // Elegimos en la iteración i como test al archivo i
      vector<Ejemplo> test = dataset[i];

      // El resto de archivos serán para entrenamiento
      vector<Ejemplo> entrenamiento;
      for(int j = 0; j < NUM_PARTICIONES; ++j)
        if (j != i) {
            vector<Ejemplo> ejemplos_entrenamiento = dataset[j];
            entrenamiento.insert(entrenamiento.end(), ejemplos_entrenamiento.begin(), ejemplos_entrenamiento.end());
        }

      vector<double> w(test[0].num_caracts);

      // Vector de pesos calculado por el algoritmo Búsqueda Local
      auto momentoInicio = std::clock();

      w = BMB(entrenamiento);

      auto momentoFin = std::clock();

      // Calculo los valores de las tasas y del fitness y los sumo a las variables de acumulación
      double tasa_clas = tasaClasificacion(test, entrenamiento, w);
      double tasa_red = tasaReduccion(w);
      double fit = fitness(tasa_clas, tasa_red);

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      double tiempo = 1000.0*(momentoFin - momentoInicio)/CLOCKS_PER_SEC;

      tasa_clas_acum += tasa_clas;
      tasa_red_acum += tasa_red;
      fit_acum += fit;
      tiempo_acum += tiempo;

      // Muestro los resultados específicos de cada iteración por pantalla
      cout << fixed << setprecision(2);
      cout << ":::" << setw(6) << (i+1) << setw(8) << ":::" << setw(15) << tasa_clas << setw(15) << ":::" << setw(13) << tasa_red;
      cout << setw(13) << ":::" << setw(7) << fit << setw(5) << "::: " << setw(9) << tiempo << std::setw(7) << ":::" << endl;      
    }

    cout << ":::" << setw(8) << "MEDIA" << setw(6) << ":::" << setw(15) << (tasa_clas_acum/NUM_PARTICIONES) << setw(15) << ":::" << setw(13) << (tasa_red_acum/NUM_PARTICIONES);
    cout << setw(13) << ":::" << setw(7) << (fit_acum/NUM_PARTICIONES) << setw(5) << "::: " << setw(9) << (tiempo_acum/NUM_PARTICIONES) << std::setw(7) << ":::" << endl;  
    cout << "....................................................................................................." << endl << endl;
  }
}

void resultadosES() {

  string nombre_archivo;

  for(int l = 0; l < NUM_ARCHIVOS; ++l) {

    switch(l) {
      case 0:
        nombre_archivo = "diabetes";
      break;

      case 1:
        nombre_archivo = "ozone-320";
      break;

      case 2:
        nombre_archivo = "spectf-heart";
      break;
    }

    // Leer y normalizar dataset
    vector< vector<Ejemplo> > dataset;
    vector<Ejemplo> particion_1 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_1.arff");
    vector<Ejemplo> particion_2 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_2.arff");
    vector<Ejemplo> particion_3 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_3.arff");
    vector<Ejemplo> particion_4 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_4.arff");
    vector<Ejemplo> particion_5 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_5.arff");

    dataset.push_back(particion_1);
    dataset.push_back(particion_2);
    dataset.push_back(particion_3);
    dataset.push_back(particion_4);
    dataset.push_back(particion_5);

    normalizarValores(dataset);

    cout << endl << endl;
    cout << "************************************ " << nombre_archivo << " (ES) ************************************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio 
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo Búsqueda Local en las diferentes particiones
    for(int i = 0; i < NUM_PARTICIONES; ++i) {
      // Elegimos en la iteración i como test al archivo i
      vector<Ejemplo> test = dataset[i];

      // El resto de archivos serán para entrenamiento
      vector<Ejemplo> entrenamiento;
      for(int j = 0; j < NUM_PARTICIONES; ++j)
        if (j != i) {
            vector<Ejemplo> ejemplos_entrenamiento = dataset[j];
            entrenamiento.insert(entrenamiento.end(), ejemplos_entrenamiento.begin(), ejemplos_entrenamiento.end());
        }

      vector<double> w(test[0].num_caracts);

      // Vector de pesos calculado por el algoritmo Búsqueda Local
      auto momentoInicio = std::clock();

      w = ES(entrenamiento);

      auto momentoFin = std::clock();

      // Calculo los valores de las tasas y del fitness y los sumo a las variables de acumulación
      double tasa_clas = tasaClasificacion(test, entrenamiento, w);
      double tasa_red = tasaReduccion(w);
      double fit = fitness(tasa_clas, tasa_red);

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      double tiempo = 1000.0*(momentoFin - momentoInicio)/CLOCKS_PER_SEC;

      tasa_clas_acum += tasa_clas;
      tasa_red_acum += tasa_red;
      fit_acum += fit;
      tiempo_acum += tiempo;

      // Muestro los resultados específicos de cada iteración por pantalla
      cout << fixed << setprecision(2);
      cout << ":::" << setw(6) << (i+1) << setw(8) << ":::" << setw(15) << tasa_clas << setw(15) << ":::" << setw(13) << tasa_red;
      cout << setw(13) << ":::" << setw(7) << fit << setw(5) << "::: " << setw(9) << tiempo << std::setw(7) << ":::" << endl;      
    }

    cout << ":::" << setw(8) << "MEDIA" << setw(6) << ":::" << setw(15) << (tasa_clas_acum/NUM_PARTICIONES) << setw(15) << ":::" << setw(13) << (tasa_red_acum/NUM_PARTICIONES);
    cout << setw(13) << ":::" << setw(7) << (fit_acum/NUM_PARTICIONES) << setw(5) << "::: " << setw(9) << (tiempo_acum/NUM_PARTICIONES) << std::setw(7) << ":::" << endl;  
    cout << "....................................................................................................." << endl << endl;
  }
}

void resultadosILS() {

  string nombre_archivo;

  for(int l = 0; l < NUM_ARCHIVOS; ++l) {

    switch(l) {
      case 0:
        nombre_archivo = "diabetes";
      break;

      case 1:
        nombre_archivo = "ozone-320";
      break;

      case 2:
        nombre_archivo = "spectf-heart";
      break;
    }

    // Leer y normalizar dataset
    vector< vector<Ejemplo> > dataset;
    vector<Ejemplo> particion_1 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_1.arff");
    vector<Ejemplo> particion_2 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_2.arff");
    vector<Ejemplo> particion_3 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_3.arff");
    vector<Ejemplo> particion_4 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_4.arff");
    vector<Ejemplo> particion_5 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_5.arff");

    dataset.push_back(particion_1);
    dataset.push_back(particion_2);
    dataset.push_back(particion_3);
    dataset.push_back(particion_4);
    dataset.push_back(particion_5);

    normalizarValores(dataset);

    cout << endl << endl;
    cout << "************************************ " << nombre_archivo << " (ILS) ************************************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio 
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo Búsqueda Local en las diferentes particiones
    for(int i = 0; i < NUM_PARTICIONES; ++i) {
      // Elegimos en la iteración i como test al archivo i
      vector<Ejemplo> test = dataset[i];

      // El resto de archivos serán para entrenamiento
      vector<Ejemplo> entrenamiento;
      for(int j = 0; j < NUM_PARTICIONES; ++j)
        if (j != i) {
            vector<Ejemplo> ejemplos_entrenamiento = dataset[j];
            entrenamiento.insert(entrenamiento.end(), ejemplos_entrenamiento.begin(), ejemplos_entrenamiento.end());
        }

      vector<double> w(test[0].num_caracts);

      // Vector de pesos calculado por el algoritmo Búsqueda Local
      auto momentoInicio = std::clock();

      w = ILS(entrenamiento);

      auto momentoFin = std::clock();

      // Calculo los valores de las tasas y del fitness y los sumo a las variables de acumulación
      double tasa_clas = tasaClasificacion(test, entrenamiento, w);
      double tasa_red = tasaReduccion(w);
      double fit = fitness(tasa_clas, tasa_red);

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      double tiempo = 1000.0*(momentoFin - momentoInicio)/CLOCKS_PER_SEC;

      tasa_clas_acum += tasa_clas;
      tasa_red_acum += tasa_red;
      fit_acum += fit;
      tiempo_acum += tiempo;

      // Muestro los resultados específicos de cada iteración por pantalla
      cout << fixed << setprecision(2);
      cout << ":::" << setw(6) << (i+1) << setw(8) << ":::" << setw(15) << tasa_clas << setw(15) << ":::" << setw(13) << tasa_red;
      cout << setw(13) << ":::" << setw(7) << fit << setw(5) << "::: " << setw(9) << tiempo << std::setw(7) << ":::" << endl;      
    }

    cout << ":::" << setw(8) << "MEDIA" << setw(6) << ":::" << setw(15) << (tasa_clas_acum/NUM_PARTICIONES) << setw(15) << ":::" << setw(13) << (tasa_red_acum/NUM_PARTICIONES);
    cout << setw(13) << ":::" << setw(7) << (fit_acum/NUM_PARTICIONES) << setw(5) << "::: " << setw(9) << (tiempo_acum/NUM_PARTICIONES) << std::setw(7) << ":::" << endl;  
    cout << "....................................................................................................." << endl << endl;
  }
}

void resultadosILS_ES() {

  string nombre_archivo;

  for(int l = 0; l < NUM_ARCHIVOS; ++l) {

    switch(l) {
      case 0:
        nombre_archivo = "diabetes";
      break;

      case 1:
        nombre_archivo = "ozone-320";
      break;

      case 2:
        nombre_archivo = "spectf-heart";
      break;
    }

    // Leer y normalizar dataset
    vector< vector<Ejemplo> > dataset;
    vector<Ejemplo> particion_1 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_1.arff");
    vector<Ejemplo> particion_2 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_2.arff");
    vector<Ejemplo> particion_3 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_3.arff");
    vector<Ejemplo> particion_4 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_4.arff");
    vector<Ejemplo> particion_5 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_5.arff");

    dataset.push_back(particion_1);
    dataset.push_back(particion_2);
    dataset.push_back(particion_3);
    dataset.push_back(particion_4);
    dataset.push_back(particion_5);

    normalizarValores(dataset);

    cout << endl << endl;
    cout << "************************************ " << nombre_archivo << " (ILS_ES) ************************************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio 
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo Búsqueda Local en las diferentes particiones
    for(int i = 0; i < NUM_PARTICIONES; ++i) {
      // Elegimos en la iteración i como test al archivo i
      vector<Ejemplo> test = dataset[i];

      // El resto de archivos serán para entrenamiento
      vector<Ejemplo> entrenamiento;
      for(int j = 0; j < NUM_PARTICIONES; ++j)
        if (j != i) {
            vector<Ejemplo> ejemplos_entrenamiento = dataset[j];
            entrenamiento.insert(entrenamiento.end(), ejemplos_entrenamiento.begin(), ejemplos_entrenamiento.end());
        }

      vector<double> w(test[0].num_caracts);

      // Vector de pesos calculado por el algoritmo Búsqueda Local
      auto momentoInicio = std::clock();

      w = ILS_ES(entrenamiento);

      auto momentoFin = std::clock();

      // Calculo los valores de las tasas y del fitness y los sumo a las variables de acumulación
      double tasa_clas = tasaClasificacion(test, entrenamiento, w);
      double tasa_red = tasaReduccion(w);
      double fit = fitness(tasa_clas, tasa_red);

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      double tiempo = 1000.0*(momentoFin - momentoInicio)/CLOCKS_PER_SEC;

      tasa_clas_acum += tasa_clas;
      tasa_red_acum += tasa_red;
      fit_acum += fit;
      tiempo_acum += tiempo;

      // Muestro los resultados específicos de cada iteración por pantalla
      cout << fixed << setprecision(2);
      cout << ":::" << setw(6) << (i+1) << setw(8) << ":::" << setw(15) << tasa_clas << setw(15) << ":::" << setw(13) << tasa_red;
      cout << setw(13) << ":::" << setw(7) << fit << setw(5) << "::: " << setw(9) << tiempo << std::setw(7) << ":::" << endl;      
    }

    cout << ":::" << setw(8) << "MEDIA" << setw(6) << ":::" << setw(15) << (tasa_clas_acum/NUM_PARTICIONES) << setw(15) << ":::" << setw(13) << (tasa_red_acum/NUM_PARTICIONES);
    cout << setw(13) << ":::" << setw(7) << (fit_acum/NUM_PARTICIONES) << setw(5) << "::: " << setw(9) << (tiempo_acum/NUM_PARTICIONES) << std::setw(7) << ":::" << endl;  
    cout << "....................................................................................................." << endl << endl;
  }
}

void resultadosVNS() {

  string nombre_archivo;

  for(int l = 0; l < NUM_ARCHIVOS; ++l) {

    switch(l) {
      case 0:
        nombre_archivo = "diabetes";
      break;

      case 1:
        nombre_archivo = "ozone-320";
      break;

      case 2:
        nombre_archivo = "spectf-heart";
      break;
    }

    // Leer y normalizar dataset
    vector< vector<Ejemplo> > dataset;
    vector<Ejemplo> particion_1 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_1.arff");
    vector<Ejemplo> particion_2 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_2.arff");
    vector<Ejemplo> particion_3 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_3.arff");
    vector<Ejemplo> particion_4 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_4.arff");
    vector<Ejemplo> particion_5 = leerFicheroARFF("../BIN/DATA/" + nombre_archivo + "_5.arff");

    dataset.push_back(particion_1);
    dataset.push_back(particion_2);
    dataset.push_back(particion_3);
    dataset.push_back(particion_4);
    dataset.push_back(particion_5);

    normalizarValores(dataset);

    cout << endl << endl;
    cout << "************************************ " << nombre_archivo << " (VNS) ************************************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio 
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo Búsqueda Local en las diferentes particiones
    for(int i = 0; i < NUM_PARTICIONES; ++i) {
      // Elegimos en la iteración i como test al archivo i
      vector<Ejemplo> test = dataset[i];

      // El resto de archivos serán para entrenamiento
      vector<Ejemplo> entrenamiento;
      for(int j = 0; j < NUM_PARTICIONES; ++j)
        if (j != i) {
            vector<Ejemplo> ejemplos_entrenamiento = dataset[j];
            entrenamiento.insert(entrenamiento.end(), ejemplos_entrenamiento.begin(), ejemplos_entrenamiento.end());
        }

      vector<double> w(test[0].num_caracts);

      // Vector de pesos calculado por el algoritmo Búsqueda Local
      auto momentoInicio = std::clock();

      w = VNS(entrenamiento);

      auto momentoFin = std::clock();

      // Calculo los valores de las tasas y del fitness y los sumo a las variables de acumulación
      double tasa_clas = tasaClasificacion(test, entrenamiento, w);
      double tasa_red = tasaReduccion(w);
      double fit = fitness(tasa_clas, tasa_red);

      // Calculo el tiempo que le ha tomado al algoritmo ejecutarse
      double tiempo = 1000.0*(momentoFin - momentoInicio)/CLOCKS_PER_SEC;

      tasa_clas_acum += tasa_clas;
      tasa_red_acum += tasa_red;
      fit_acum += fit;
      tiempo_acum += tiempo;

      // Muestro los resultados específicos de cada iteración por pantalla
      cout << fixed << setprecision(2);
      cout << ":::" << setw(6) << (i+1) << setw(8) << ":::" << setw(15) << tasa_clas << setw(15) << ":::" << setw(13) << tasa_red;
      cout << setw(13) << ":::" << setw(7) << fit << setw(5) << "::: " << setw(9) << tiempo << std::setw(7) << ":::" << endl;      
    }

    cout << ":::" << setw(8) << "MEDIA" << setw(6) << ":::" << setw(15) << (tasa_clas_acum/NUM_PARTICIONES) << setw(15) << ":::" << setw(13) << (tasa_red_acum/NUM_PARTICIONES);
    cout << setw(13) << ":::" << setw(7) << (fit_acum/NUM_PARTICIONES) << setw(5) << "::: " << setw(9) << (tiempo_acum/NUM_PARTICIONES) << std::setw(7) << ":::" << endl;  
    cout << "....................................................................................................." << endl << endl;
  }
}
