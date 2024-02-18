// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica Alternativa al Examen: Metaheurística Leaders and Followers
// Problema: APC

#include "pAlternativa.h"

using namespace std;

// *************************************************************************
// ******************* Búsqueda Local de Baja Intensidad *******************
// *************************************************************************


int busquedaLocalBajaIntensidad(const vector<Ejemplo>& entrenamiento, LF& individuo) {

  normal_distribution<double> distribucion_normal(0.0, SIGMA);

  vector<int> indices;     // Se utiliza para seleccionar la componente de w a mutar
  double mejor_fitness;

  int num_iteraciones = 0;

  // Inicializamos el vector de índices
  for(int i = 0; i < individuo.w.size(); ++i)
    indices.push_back(i);

  Random::shuffle(indices);

  mejor_fitness = individuo.fitness;

  // Búsqueda en el vecindario del primero mejor
  while(num_iteraciones < individuo.w.size() * COEF_MAX_VECINOS_BAJA_INTENSIDAD ) {

    // Seleccionamos la componente de w para mutar
    int comp_mutada = indices[num_iteraciones % individuo.w.size()];

    // Mutación Normal de w
    LF individuo_mutado = individuo;
    individuo_mutado.w[comp_mutada] += Random::get(distribucion_normal);

    // Truncamos el peso de la componente mutada si fuera necesario
    if(individuo_mutado.w[comp_mutada] > 1) 
      individuo_mutado.w[comp_mutada] = 1;
    else if(individuo_mutado.w[comp_mutada] < 0) 
      individuo_mutado.w[comp_mutada] = 0;

    // Vemos si el fitness dado por individuo_mutado mejora mejor_fitness, en cuyo caso actualizamos dicho valor
    individuo_mutado.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, individuo_mutado.w), tasaReduccion(individuo_mutado.w));

    num_iteraciones++;

    if(individuo_mutado.fitness > mejor_fitness) {
      individuo = individuo_mutado;
      mejor_fitness = individuo_mutado.fitness;
    }

    // Actualizamos el vector de índices si ya se ha recorrido entero sin encontrar mejora
    if(num_iteraciones % individuo.w.size() == 0)
      Random::shuffle(indices);
  }

  return num_iteraciones;
}



// *************************************************************************
// ************************* Funciones Auxiliares **************************
// *************************************************************************

PoblacionLF inicializarPoblacionLF(const vector<Ejemplo>& entrenamiento) {
  
  PoblacionLF p;

  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);

  int tamanio_poblacion = entrenamiento[0].num_caracts;

  for(int i = 0; i < tamanio_poblacion; ++i) {
    LF individuo;

    individuo.w.resize(tamanio_poblacion);

    for(int j = 0; j < individuo.w.size(); ++j)
      individuo.w[j] = Random::get(distribucion_uniform_real);  // Inicializar vector de pesos

    individuo.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, individuo.w), tasaReduccion(individuo.w));    // Calcular fitness

    p.insert(individuo);
  }

  return p;
}

LF createTrial(const LF& leader, const LF& follower, const vector<Ejemplo>& entrenamiento) {

  LF trial;

  trial.w.resize(leader.w.size());

  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);

  double epsilon = Random::get(distribucion_uniform_real);
  
  for(int i = 0; i < trial.w.size(); ++i) {
    trial.w[i] = follower.w[i] + epsilon * 2 * (leader.w[i] - follower.w[i]);

    if(trial.w[i] > 1.0)
      trial.w[i] = 1.0;
    if(trial.w[i] < 0.0)
      trial.w[i] = 0.0;
  }

  trial.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, trial.w), tasaReduccion(trial.w));
  
  return trial;
}

LF createTrialModificado(const LF& leader, const LF& follower, const vector<Ejemplo>& entrenamiento, double mutacion) {

  LF trial;

  trial.w.resize(leader.w.size());

  normal_distribution<double> distribucion_normal(0.0, mutacion);

  for(int i = 0; i < trial.w.size(); ++i) {
    double promedio = (leader.w[i] + follower.w[i]) / 2.0;
    double delta = Random::get(distribucion_normal);
    
    trial.w[i] = promedio + delta;

    if(trial.w[i] > 1.0)
      trial.w[i] = 1.0;
    if(trial.w[i] < 0.0)
      trial.w[i] = 0.0;
  }

  trial.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, trial.w), tasaReduccion(trial.w));

  return trial;
}





double calcularMediana(const PoblacionLF& p) {

  vector<double> v_fitness(p.size());

  for(int i = 0; i < p.size(); ++i) {
    auto it = p.begin();
    LF individuo = *it;
    v_fitness.push_back(individuo.fitness);
  }

  double mediana;

  if(v_fitness.size() % 2 == 0)
    mediana  = ( v_fitness[v_fitness.size() / 2] + v_fitness[ (v_fitness.size() / 2) - 1] ) / 2.0;
  else
    mediana = v_fitness[v_fitness.size() / 2];
  
  return mediana;
}

PoblacionLF mergePopulations(const PoblacionLF& leaders, const PoblacionLF& followers) {
  PoblacionLF p;

  PoblacionLF copia_leaders = leaders;
  PoblacionLF copia_followers = followers;

  auto it = copia_leaders.rbegin();
  LF mejor_leader = *it;
  p.insert(mejor_leader);
  copia_leaders.erase(std::next(it).base()); 

  // Torneo binario para obtener los n-1 individuos restantes de la nueva población
  while(p.size() < leaders.size()) {
    LF ganador;

    if (!copia_leaders.empty() && !copia_followers.empty()) {
      uniform_int_distribution<int> distribucion_uniform_int_leaders(0, copia_leaders.size() - 1);
      uniform_int_distribution<int> distribucion_uniform_int_followers(0, copia_followers.size() - 1);

      auto it_l = copia_leaders.begin();
      auto it_f = copia_followers.begin();

      advance(it_l, Random::get(distribucion_uniform_int_leaders));
      advance(it_f, Random::get(distribucion_uniform_int_followers));

      if (it_l->fitness > it_f->fitness)
        ganador = *it_l;
      else
        ganador = *it_f;

      p.insert(ganador);

      copia_leaders.erase(it_l);
      copia_followers.erase(it_f);
    } 
    else if (!copia_leaders.empty()) {
      auto it_l = copia_leaders.begin();
      ganador = *it_l;

      p.insert(ganador);
      copia_leaders.erase(it_l);
    } 
    else if (!copia_followers.empty()) {
      auto it_f = copia_followers.begin();
      ganador = *it_f;

      p.insert(ganador);
      copia_followers.erase(it_f);
    }
  }

  return p;
}




// *************************************************************************
// ***************** Algoritmos Leaders and Followers (LFs) ****************
// *************************************************************************

vector<double> LeadersAndFollowers(const vector<Ejemplo>& entrenamiento) {

  int num_iteraciones = 0;

  PoblacionLF leaders;
  PoblacionLF followers;

  vector<double> w(entrenamiento[0].num_caracts);

  int tamanio_poblacion = w.size();

  leaders = inicializarPoblacionLF(entrenamiento);
  followers = inicializarPoblacionLF(entrenamiento);

  while(num_iteraciones < MAX_ITER_LEADERS_AND_FOLLOWERS) {

    for(int i = 0; i < tamanio_poblacion; ++i) {
      auto it_l = leaders.rbegin(); 
      advance(it_l, i);
      LF leader = *it_l;

      auto it_f = followers.rbegin(); 
      advance(it_f, i);
      LF follower = *it_f;

      LF trial = createTrial(leader, follower, entrenamiento);

      if(trial.fitness > follower.fitness)
        follower = trial;
    }

    if( calcularMediana(leaders) > calcularMediana(followers) ) {
      leaders = mergePopulations(leaders, followers);
      followers = inicializarPoblacionLF(entrenamiento);
    }

    num_iteraciones++;
  }

  w = leaders.rbegin()->w;

  return w;
}

vector<double> LeadersAndFollowersBL(const vector<Ejemplo>& entrenamiento) {

  int num_iteraciones = 0;

  PoblacionLF leaders;
  PoblacionLF followers;

  vector<double> w(entrenamiento[0].num_caracts);

  int tamanio_poblacion = w.size();

  leaders = inicializarPoblacionLF(entrenamiento);
  followers = inicializarPoblacionLF(entrenamiento);

  while(num_iteraciones < MAX_ITER_LEADERS_AND_FOLLOWERS) {

    for(int i = 0; i < tamanio_poblacion; ++i) {
      auto it_l = leaders.rbegin(); 
      advance(it_l, i);
      LF leader = *it_l;

      auto it_f = followers.rbegin(); 
      advance(it_f, i);
      LF follower = *it_f;

      LF trial = createTrial(leader, follower, entrenamiento);

      num_iteraciones += busquedaLocalBajaIntensidad(entrenamiento, trial);

      if(trial.fitness > follower.fitness)
        follower = trial;
    }

    if( calcularMediana(leaders) > calcularMediana(followers) ) {
      leaders = mergePopulations(leaders, followers);
      followers = inicializarPoblacionLF(entrenamiento);
    }

    num_iteraciones++;
  }

  w = leaders.rbegin()->w;

  return w;
}

vector<double> LeadersAndFollowersModificado(const vector<Ejemplo>& entrenamiento) {

  int num_iteraciones = 0;

  PoblacionLF leaders;
  PoblacionLF followers;

  vector<double> w(entrenamiento[0].num_caracts);

  int tamanio_poblacion = w.size();

  leaders = inicializarPoblacionLF(entrenamiento);
  followers = inicializarPoblacionLF(entrenamiento);

  double mutacion_inicial = 0.1;
  double mutacion_minima = 0.01;
  double factor_ajuste = 0.9;

  double mutacion = mutacion_inicial;

  while(num_iteraciones < MAX_ITER_LEADERS_AND_FOLLOWERS) {

    for(int i = 0; i < tamanio_poblacion; ++i) {
      auto it_l = leaders.begin(); 
      advance(it_l, i);
      LF leader = *it_l;

      auto it_f = followers.rbegin(); 
      advance(it_f, i);
      LF follower = *it_f;

      LF trial = createTrialModificado(leader, follower, entrenamiento, mutacion);

      if(trial.fitness > follower.fitness)
        follower = trial;
    }

    if( calcularMediana(leaders) > calcularMediana(followers) ) {
      leaders = mergePopulations(leaders, followers);
      followers = inicializarPoblacionLF(entrenamiento);
    }

    mutacion *= factor_ajuste;

    if(mutacion < mutacion_minima)
      mutacion = mutacion_minima;

    num_iteraciones++;
  }

  w = leaders.rbegin()->w;

  return w;
}




// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

void resultadosLeadersAndFollowers() {

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
    cout << "************************************ " << nombre_archivo << " (LeadersAndFollowers) ************************************************" << endl;

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

      w = LeadersAndFollowers(entrenamiento);

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

void resultadosLeadersAndFollowersBL() {

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
    cout << "************************************ " << nombre_archivo << " (LeadersAndFollowersBL) ************************************************" << endl;

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

      w = LeadersAndFollowersBL(entrenamiento);

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

void resultadosLeadersAndFollowersModificado() {

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
    cout << "************************************ " << nombre_archivo << " (LeadersAndFollowersModificado) ************************************************" << endl;

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

      w = LeadersAndFollowersModificado(entrenamiento);

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
