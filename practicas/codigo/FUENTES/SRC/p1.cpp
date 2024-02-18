// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
// Problema: APC

#include "p1.h"

using namespace std;

// *************************************************************************
// *************************** Clasificador 1-NN ***************************
// *************************************************************************

string clasificador1NNPesos(const Ejemplo& e, const vector<Ejemplo>& entrenamiento, const vector<double>& w) {

  double distancia;
  double distancia_min = numeric_limits<double>::max();
  string clase;

  for(int i = 0; i < entrenamiento.size(); ++i) {
    distancia = distanciaEuclideaPesos(e, entrenamiento[i], w);
     
    if(distancia < distancia_min) {
      distancia_min = distancia;
      clase = entrenamiento[i].categoria;
    }
  }

  return clase;
}

// *************************************************************************
// ***************************** Greedy RELIEF *****************************
// *************************************************************************

Ejemplo amigoMasCercano(const Ejemplo& e,const vector<Ejemplo>& entrenamiento) {

  Ejemplo amigo_mas_cercano;
  double distancia;
  double distancia_min = numeric_limits<double>::max();
  
  for(int i = 0; i < entrenamiento.size(); ++i)
    if( (e.val_caracts != entrenamiento[i].val_caracts) && (entrenamiento[i].categoria == e.categoria) ) {
      distancia = distanciaEuclidea(e,entrenamiento[i]);
      
      if(distancia < distancia_min) {
        distancia_min = distancia;
        amigo_mas_cercano = entrenamiento[i];
      }
    }

  return amigo_mas_cercano;
}

Ejemplo enemigoMasCercano(const Ejemplo& e,const vector<Ejemplo>& entrenamiento) {

  Ejemplo enemigo_mas_cercano;
  double distancia;
  double distancia_min = numeric_limits<double>::max();
  
  for(int i = 0; i < entrenamiento.size(); ++i)
    if(entrenamiento[i].categoria != e.categoria ) {
      distancia = distanciaEuclidea(e,entrenamiento[i]);
      
      if(distancia < distancia_min) {
        distancia_min = distancia;
        enemigo_mas_cercano = entrenamiento[i];
      }
          
      }

  return enemigo_mas_cercano;
}

vector<double> greedy(const vector<Ejemplo>& entrenamiento) {

  vector<double> w(entrenamiento[0].num_caracts);
  inicializarVectorPesos(w);

  for(int i = 0; i < entrenamiento.size(); ++i) {
    Ejemplo enemigo_mas_cercano = enemigoMasCercano(entrenamiento[i],entrenamiento);
    Ejemplo amigo_mas_cercano = amigoMasCercano(entrenamiento[i],entrenamiento);

    for(int j = 0; j < w.size(); ++j)
      w[j] = w[j] + abs(entrenamiento[i].val_caracts[j] - enemigo_mas_cercano.val_caracts[j]) - abs(entrenamiento[i].val_caracts[j] - amigo_mas_cercano.val_caracts[j]);
  }

  float w_max = *max_element(w.begin(),w.end());

  for(int j = 0; j < w.size(); ++j) {
    if(w[j] < 0)
      w[j] = 0.0;
    else
      w[j] = w[j]/w_max;
  }

  return w;
}



// *************************************************************************
// ************************** Búsqueda Local (BL) **************************
// *************************************************************************

vector<double> busquedaLocal(const vector<Ejemplo>& entrenamiento) {

  normal_distribution<double> distribucion_normal(0.0, SIGMA);
  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);

  vector<double> w(entrenamiento[0].num_caracts);
  vector<int> indices;            // Se utiliza para seleccionar la componente de w a mutar
  double mejor_fitness;

  int num_iteraciones = 0;
  int num_vecinos = 0;

  bool hay_mejora = false;

  // Inicializamos el vector de índices y el vector de pesos (este último se hace aleatoriamente)
  for(int i = 0; i < w.size(); ++i) {
    indices.push_back(i);
    w[i] = Random::get(distribucion_uniform_real);
  }

  Random::shuffle(indices);

  mejor_fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, w), tasaReduccion(w));

  // Búsqueda en el vecindario del primero mejor
  while( (num_iteraciones < MAX_ITER) && (num_vecinos < w.size() * COEF_MAX_VECINOS) ) {

    // Seleccionamos la componente de w para mutar
    int comp_mutada = indices[num_iteraciones % w.size()];

    // Mutación Normal de w
    vector<double> w_mutado = w;
    w_mutado[comp_mutada] += Random::get(distribucion_normal);

    // Truncamos el peso de la componente mutada si fuera necesario
    if(w_mutado[comp_mutada] > 1) 
      w_mutado[comp_mutada] = 1;
    else if(w_mutado[comp_mutada] < 0) 
      w_mutado[comp_mutada] = 0;

    // Vemos si el fitness dado por w_mutado mejora mejor_fitness, en cuyo caso actualizamos dicho valor
    double fitness_actual = fitness(tasaClasificacionLeaveOneOut(entrenamiento, w), tasaReduccion(w_mutado));

    num_iteraciones++;

    if(fitness_actual > mejor_fitness) {
      num_vecinos = 0;
      w = w_mutado;
      mejor_fitness = fitness_actual;
      hay_mejora = true;
    }
    else
      num_vecinos++;

    // Actualizamos el vector de índices si ha habido mejora o ya se ha recorrido entero sin encontrar mejora
    if(num_iteraciones % w.size() == 0 || hay_mejora) {
      Random::shuffle(indices);
      hay_mejora = false;
    }

  }

  return w;
}



// *************************************************************************
// ************************* Función de Evaluación *************************
// *************************************************************************

double tasaClasificacion(const vector<Ejemplo>& test, const vector<Ejemplo>& entrenamiento, const vector<double>& w) {

  int num_instancias_bien_clas = 0;

  for(int i = 0; i < test.size(); ++i)
    if(clasificador1NNPesos(test[i], entrenamiento, w) == test[i].categoria)
      num_instancias_bien_clas++;

  return (100.0 * double(num_instancias_bien_clas) ) / double(test.size());
}

double tasaClasificacionLeaveOneOut(const vector<Ejemplo>& entrenamiento,const vector<double>& w) {

  Ejemplo ejemplo;
  vector<Ejemplo> aux = entrenamiento;
  int num_instancias_bien_clas = 0;

  for(int i = 0; i < aux.size(); ++i) {
    ejemplo = aux[i];
    aux.erase(aux.begin()+i);

    if(clasificador1NNPesos(ejemplo, aux, w) == ejemplo.categoria)
      num_instancias_bien_clas++;
    
    aux.insert(aux.begin()+i,ejemplo);
  }

  return (100.0 * double(num_instancias_bien_clas) ) / double(entrenamiento.size());
}

double tasaReduccion(const vector<double>& w) {

  int num_caracts_descartadas = 0;

  for(int i = 0; i < w.size(); ++i)
    if(w[i] < 0.1)
      num_caracts_descartadas++;

  return (100.0 * double(num_caracts_descartadas) ) / double(w.size());
}

double fitness(double tasa_clas, double tasa_red) {

  return (ALPHA * tasa_clas) + ( (1.0 - ALPHA) * tasa_red );
}

    

// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

void resultados1NN() {

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
    cout << "************************************ " << nombre_archivo << " (1-NN) ************************************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio 
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo 1-NN en las diferentes particiones
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

      // Vector de pesos para el algoritmo 1-NN
      for(int j = 0; j < w.size(); ++j)
        w[j] = 1.0;

      auto momentoInicio = std::clock();

      // Calculo los valores de las tasas y del fitness, donde se ejecuta el algoritmo 1-NN, y los sumo a las variables de acumulación
      double tasa_clas = tasaClasificacion(test, entrenamiento, w);
      double tasa_red = tasaReduccion(w);
      double fit = fitness(tasa_clas, tasa_red);

      auto momentoFin = std::clock();

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

void resultadosGreedy() {

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
    cout << "************************************ " << nombre_archivo << " (greedy) ************************************************" << endl;

    cout << endl << "....................................................................................................." << endl;
    cout << "::: Particion ::: Tasa de Clasificacion (%) ::: Tasa de Reduccion (%) ::: Fitness ::: Tiempo (ms) :::" << endl;
    cout << "....................................................................................................." << endl;

    // Declaración de los resultados que vamos a acumular para mostrar finalmente un resultado medio 
    double tasa_clas_acum = 0.0;
    double tasa_red_acum = 0.0;
    double fit_acum = 0.0;
    double tiempo_acum = 0.0;

    // Ejecución del algoritmo Greedy RELIEF en las diferentes particiones
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

      // Vector de pesos calculado por el algoritmo Greedy RELIEF
      auto momentoInicio = std::clock();

      w = greedy(entrenamiento);

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

void resultadosBusquedaLocal() {

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
    cout << "************************************ " << nombre_archivo << " (busquedaLocal) ************************************************" << endl;

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

      w = busquedaLocal(entrenamiento);

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
