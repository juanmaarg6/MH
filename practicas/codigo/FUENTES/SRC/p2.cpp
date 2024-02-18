// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
// Problema: APC

#include "p2.h"

using namespace std;

// *************************************************************************
// ******************* Búsqueda Local de Baja Intensidad *******************
// *************************************************************************

int busquedaLocalBajaIntensidad(const vector<Ejemplo>& entrenamiento, Cromosoma& c) {

  normal_distribution<double> distribucion_normal(0.0, SIGMA);

  vector<int> indices;     // Se utiliza para seleccionar la componente de w a mutar
  double mejor_fitness;

  int num_iteraciones = 0;

  // Inicializamos el vector de índices
  for(int i = 0; i < c.w.size(); ++i)
    indices.push_back(i);

  Random::shuffle(indices);

  mejor_fitness = c.fitness;

  // Búsqueda en el vecindario del primero mejor
  while(num_iteraciones < c.w.size() * COEF_MAX_VECINOS_BAJA_INTENSIDAD ) {

    // Seleccionamos la componente de w para mutar
    int comp_mutada = indices[num_iteraciones % c.w.size()];

    // Mutación Normal de w
    Cromosoma c_mutado = c;
    c_mutado.w[comp_mutada] += Random::get(distribucion_normal);

    // Truncamos el peso de la componente mutada si fuera necesario
    if(c_mutado.w[comp_mutada] > 1) 
      c_mutado.w[comp_mutada] = 1;
    else if(c_mutado.w[comp_mutada] < 0) 
      c_mutado.w[comp_mutada] = 0;

    // Vemos si el fitness dado por c_mutado mejora mejor_fitness, en cuyo caso actualizamos dicho valor
    c_mutado.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, c_mutado.w), tasaReduccion(c_mutado.w));

    num_iteraciones++;

    if(c_mutado.fitness > mejor_fitness) {
      c = c_mutado;
      mejor_fitness = c_mutado.fitness;
    }

    // Actualizamos el vector de índices si ya se ha recorrido entero sin encontrar mejora
    if(num_iteraciones % c.w.size() == 0)
      Random::shuffle(indices);
  }

  return num_iteraciones;
}



// *************************************************************************
// ************************* Operadores Genéticos **************************
// *************************************************************************

void inicializarPoblacion(Poblacion& p, int num_cromosomas, int tamanio, const vector<Ejemplo>& entrenamiento) {
  
  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);

  for(int i = 0; i < num_cromosomas; ++i) {
    Cromosoma c;
    c.w.resize(tamanio);

    for(int j = 0; j < tamanio; ++j)
      c.w[j] = Random::get(distribucion_uniform_real);

    c.fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, c.w), tasaReduccion(c.w));
    p.insert(c);
  }
}

Cromosoma seleccion(const Poblacion& p) {
  
  uniform_int_distribution<int> distribucion_uniform_int(0, p.size() - 1);

  // Iteradores a dos cromosomas randoms de la población
  auto p1 = p.begin();
  auto p2 = p.begin();
  advance(p1, Random::get(distribucion_uniform_int));
  advance(p2, Random::get(distribucion_uniform_int));

  return p1->fitness > p2->fitness ? *p1 : *p2;
}

pair<Cromosoma, Cromosoma> cruceBLX(const Cromosoma& c1, const Cromosoma& c2) {
  
  Cromosoma h1, h2;

  h1.w.resize(c1.w.size());
  h2.w.resize(c1.w.size());

  for(int i = 0; i < c1.w.size(); ++i) {
    float c_min = min(c1.w[i], c2.w[i]);
    float c_max = max(c1.w[i], c2.w[i]);
    float dif = c_max - c_min;

    uniform_real_distribution<float> distribucion_uniform_real(c_min - dif * ALPHA_BLX, c_max + dif * ALPHA_BLX);

    // NOTA: Si c1.w[i] == c2.w[i], entonces h1.w[i] = h2.w[i] = c1.w[i]

    h1.w[i] = Random::get(distribucion_uniform_real);
    h2.w[i] = Random::get(distribucion_uniform_real);

    // Truncamiento
    if(h1.w[i] < 0) 
      h1.w[i] = 0.0;
    if(h1.w[i] > 1) 
      h1.w[i] = 1.0;
    if(h2.w[i] < 0) 
      h2.w[i] = 0.0;
    if(h2.w[i] > 1) 
      h2.w[i] = 1.0;
  }

  h1.fitness = -1.0;
  h2.fitness = -1.0;

  return make_pair(h1, h2);
}

Cromosoma cruceAritmetico(const Cromosoma& c1, const Cromosoma& c2) {
  
  Cromosoma h;

  h.w.resize(c1.w.size());

  for(int i = 0; i < c1.w.size(); ++i)
    h.w[i] = (c1.w[i] + c2.w[i]) / 2.0;

  h.fitness = -1.0;

  return h;
}

void mutacion(Cromosoma& c, int comp) {
  
  normal_distribution<double> distribucion_normal(0.0, SIGMA);

  c.w[comp] += Random::get(distribucion_normal);
  c.fitness = -1.0;

  // Truncamiento
  if(c.w[comp] < 0) 
    c.w[comp] = 0.0;
  if(c.w[comp] > 1) 
    c.w[comp] = 1.0;
}

int mutacionesEsperadas(int num_genes_individuo, int total_genes) {

  float mutaciones_esperadas = float(PROBABILIDAD_MUTACION / num_genes_individuo) * total_genes;

  if(mutaciones_esperadas <= 1.0)
    return 1;

  float resto = modf(mutaciones_esperadas, &mutaciones_esperadas);

  uniform_real_distribution<double> distribucion_uniform_real(0.0, 1.0);
  double valor_aleatorio = Random::get(distribucion_uniform_real);

  if(valor_aleatorio <= resto)
    mutaciones_esperadas++;

  return mutaciones_esperadas;
}



// *************************************************************************
// *************** Algoritmos Genéticos Generacionales (AGGs) **************
// *************************************************************************

vector<double> AGG_BLX(const vector<Ejemplo> entrenamiento) {

  Poblacion p;
  Poblacion::reverse_iterator mejor_padre;  // Elitismo

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;
  int num_generaciones = 1;
  int num_genes_individuo = w.size();
  int num_total_genes = w.size() * TAMANIO_AGG;
  int num_cruces_esperados = PROBABILIDAD_CRUCE * (TAMANIO_AGG / 2);  // Número de cruces esperados

  uniform_int_distribution<int> distribucion_uniform_int(0, num_total_genes - 1);

  // 1. Construir y evaluar una población inicial
  inicializarPoblacion(p, TAMANIO_AGG, w.size(), entrenamiento);
  num_iteraciones += TAMANIO_AGG;

  while(num_iteraciones < MAX_ITER) {

    PoblacionIntermedia p_intermedia;
    Poblacion nueva_p;

    mejor_padre = p.rbegin();  // Guardamos el mejor padre para elitismo

    // 2. Seleccionar población intermedia (ya evaluada)
    p_intermedia.resize(TAMANIO_AGG);

    for(int i = 0; i < TAMANIO_AGG; ++i)
      p_intermedia[i] = seleccion(p);

    // 3. Cruzar la población intermedia con el operador BLX
    for(int i = 0; i < 2 * num_cruces_esperados; i += 2) {
      auto descendientes = cruceBLX(p_intermedia[i], p_intermedia[i+1]);
      p_intermedia[i] = descendientes.first;
      p_intermedia[i+1] = descendientes.second;
    }

    // 4. Mutar la población intermedia
    set<int> mutados;
    int num_mutaciones_esperadas = mutacionesEsperadas(num_genes_individuo, num_total_genes);

    for(int i = 0; i < num_mutaciones_esperadas; ++i) {
      int comp;

      // Seleccionar componente a mutar, sin repetición
      while(mutados.size() == i) {
        comp = Random::get(distribucion_uniform_int);
        mutados.insert(comp);
      }

      int comp_seleccionada = comp / w.size();
      int gen_seleccionado = comp % w.size();

      mutacion(p_intermedia[comp_seleccionada], gen_seleccionado);
    }

    // 5. Evaluar, reemplazar la población original y aplicar elitismo
    for(int i = 0; i < TAMANIO_AGG; ++i) {

      if(p_intermedia[i].fitness == -1.0) {
        p_intermedia[i].fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, p_intermedia[i].w), tasaReduccion(p_intermedia[i].w));
        num_iteraciones++;
      }

      nueva_p.insert(p_intermedia[i]);
    }

    auto mejor_padre_actual = nueva_p.rbegin();

    if(mejor_padre_actual->fitness < mejor_padre->fitness) {
      // Reemplazar el peor cromosoma de la población intermedia
      nueva_p.erase(nueva_p.begin());
      nueva_p.insert(*mejor_padre);
    }

    // 6. Reemplazar completamente la población previa (nueva generación)
    p = nueva_p;
    num_generaciones++;
  }

  // Elegir como solución el mejor cromosoma
  w = p.rbegin()->w;

  return w;
}

vector<double> AGG_Arit(const vector<Ejemplo> entrenamiento) {

  Poblacion p;
  Poblacion::reverse_iterator mejor_padre;  // Elitismo

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;
  int num_generaciones = 1;
  int num_genes_individuo = w.size();
  int num_total_genes = w.size() * TAMANIO_AGG;
  int num_cruces_esperados = PROBABILIDAD_CRUCE * (TAMANIO_AGG / 2);  // Número de cruces esperados

  uniform_int_distribution<int> distribucion_uniform_int(0, num_total_genes - 1);

  // 1. Construir y evaluar una población inicial
  inicializarPoblacion(p, TAMANIO_AGG, w.size(), entrenamiento);
  num_iteraciones += TAMANIO_AGG;

  while(num_iteraciones < MAX_ITER) {

    PoblacionIntermedia p_intermedia;
    Poblacion nueva_p;

    mejor_padre = p.rbegin();  // Guardamos el mejor padre para elitismo

    // 2. Seleccionar población intermedia (ya evaluada)
    // NOTA: Seleccionamos 2 * TAMANIO_AGG porque solo se genera un descendiente por cada dos padres
    p_intermedia.resize(2 * TAMANIO_AGG);

    for(int i = 0; i < 2 * TAMANIO_AGG; ++i)
      p_intermedia[i] = seleccion(p);

    // 3. Cruzar la población intermedia con el operador artimético
    for(int i = 0; i < 2 * num_cruces_esperados; ++i)
      p_intermedia[i] = cruceAritmetico(p_intermedia[i], p_intermedia[2 * TAMANIO_AGG - i - 1]);

    // 4. Mutar la población intermedia
    set<int> mutados;
    int num_mutaciones_esperadas = mutacionesEsperadas(num_genes_individuo, num_total_genes);

    for(int i = 0; i < num_mutaciones_esperadas; ++i) {
      int comp;

      // Seleccionar componente a mutar, sin repetición
      while(mutados.size() == i) {
        comp = Random::get(distribucion_uniform_int);
        mutados.insert(comp);
      }

      int comp_seleccionada = comp / w.size();
      int gen_seleccionado = comp % w.size();

      mutacion(p_intermedia[comp_seleccionada], gen_seleccionado);
    }

    // 5. Evaluar, reemplazar la población original y aplicar elitismo
    for(int i = 0; i < TAMANIO_AGG; ++i) {

      if(p_intermedia[i].fitness == -1.0) {
        p_intermedia[i].fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, p_intermedia[i].w), tasaReduccion(p_intermedia[i].w));
        num_iteraciones++;
      }

      nueva_p.insert(p_intermedia[i]);
    }

    auto mejor_padre_actual = nueva_p.rbegin();

    if(mejor_padre_actual->fitness < mejor_padre->fitness) {
      // Reemplazar el peor cromosoma de la población intermedia
      nueva_p.erase(nueva_p.begin());
      nueva_p.insert(*mejor_padre);
    }

    // 6. Reemplazar completamente la población previa (nueva generación)
    p = nueva_p;
    num_generaciones++;
  }

  // Elegir como solución el mejor cromosoma
  w = p.rbegin()->w;

  return w;
}



// *************************************************************************
// *************** Algoritmos Genéticos Estacionarios (AGEs) ***************
// *************************************************************************

vector<double> AGE_BLX(const vector<Ejemplo> entrenamiento) {

  Poblacion p;

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;
  int num_generaciones = 1;
  int num_genes = w.size();
  int num_cruces_esperados = 1.0 * (TAMANIO_AGE / 2);  // Número de cruces esperados (PROBABILIDAD_CRUCE = 1.0)
  float prob_mut = PROBABILIDAD_MUTACION * TAMANIO_AGE;

  uniform_int_distribution<int> distribucion_uniform_int(0, num_genes - 1);

  // 1. Construir y evaluar una población inicial
  inicializarPoblacion(p, TAMANIO_AGG, w.size(), entrenamiento);
  num_iteraciones += TAMANIO_AGG;

  while(num_iteraciones < MAX_ITER) {

    PoblacionIntermedia p_intermedia;
    Poblacion nueva_p;

    // 2. Seleccionar población intermedia (ya evaluada)
    p_intermedia.resize(TAMANIO_AGE);

    for(int i = 0; i < TAMANIO_AGE; ++i)
      p_intermedia[i] = seleccion(p);

    // 3. Cruzar la población intermedia con el operador BLX
    for(int i = 0; i < 2 * num_cruces_esperados; i += 2) {
      auto descendientes = cruceBLX(p_intermedia[i], p_intermedia[i+1]);
      p_intermedia[i] = descendientes.first;
      p_intermedia[i+1] = descendientes.second;
    }

    // 4. Mutar la población intermedia
    uniform_real_distribution<double> distribucion_uniform_real(0, 1.0);

    for(int i = 0; i < TAMANIO_AGE; ++i)
      if(Random::get(distribucion_uniform_real) <= prob_mut) {
        int gen_seleccionado = Random::get(distribucion_uniform_int);
        mutacion(p_intermedia[i], gen_seleccionado);
      }

    // 5. Evaluar, reemplazar la población original y aplicar elitismo
    for(int i = 0; i < TAMANIO_AGE; ++i) {
      p_intermedia[i].fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, p_intermedia[i].w), tasaReduccion(p_intermedia[i].w));
      num_iteraciones++;
      nueva_p.insert(p_intermedia[i]);
    }

    // 6. Cambiar la población anterior (nueva generación)
    auto peor_cromosoma = p.begin();
    auto segundo_peor_cromosoma = ++p.begin();
    auto mejor_descendiente_actual = nueva_p.rbegin();
    auto segundo_mejor_descendiente_actual = ++nueva_p.rbegin();

    // NOTA: Este esquema de reemplazamiento solo es válido cuando TAMANIO_AGE == 2

    // Caso 1: Ambos descendientes sobreviven
    if(segundo_mejor_descendiente_actual->fitness > segundo_peor_cromosoma->fitness) {
      p.erase(segundo_peor_cromosoma);
      p.erase(p.begin());
      p.insert(*segundo_mejor_descendiente_actual);
      p.insert(*mejor_descendiente_actual);
    }
    // Caso 2: Solo el mejor descendiente sobrevive
    else if (mejor_descendiente_actual->fitness > peor_cromosoma->fitness) {
      p.erase(peor_cromosoma);
      p.insert(*mejor_descendiente_actual);
    }

    num_generaciones++;
  }

  // Elegir como solución el mejor cromosoma
  w = p.rbegin()->w;

  return w;
}

vector<double> AGE_Arit(const vector<Ejemplo> entrenamiento) {

  Poblacion p;

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;
  int num_generaciones = 1;
  int num_genes = w.size();
  int num_cruces_esperados = 1.0 * (TAMANIO_AGE / 2);  // Número de cruces esperados (PROBABILIDAD_CRUCE = 1.0)
  float prob_mut = PROBABILIDAD_MUTACION * TAMANIO_AGE;

  uniform_int_distribution<int> distribucion_uniform_int(0, num_genes - 1);

  // 1. Construir y evaluar una población inicial
  inicializarPoblacion(p, TAMANIO_AGG, w.size(), entrenamiento);
  num_iteraciones += TAMANIO_AGG;

  while(num_iteraciones < MAX_ITER) {

    PoblacionIntermedia p_intermedia;
    Poblacion nueva_p;

    // 2. Seleccionar población intermedia (ya evaluada)
    p_intermedia.resize(2 * TAMANIO_AGE);

    for(int i = 0; i < 2 * TAMANIO_AGE; ++i)
      p_intermedia[i] = seleccion(p);

    // 3. Cruzar la población intermedia con el operador aritmético
    for(int i = 0; i < 2 * num_cruces_esperados; ++i)
      p_intermedia[i] = cruceAritmetico(p_intermedia[i], p_intermedia[2 * TAMANIO_AGE - i - 1]);

    // 4. Mutar la población intermedia
    uniform_real_distribution<double> distribucion_uniform_real(0, 1.0);

    for(int i = 0; i < TAMANIO_AGE; ++i)
      if(Random::get(distribucion_uniform_real) <= prob_mut) {
        int gen_seleccionado = Random::get(distribucion_uniform_int);
        mutacion(p_intermedia[i], gen_seleccionado);
      }

    // 5. Evaluar, reemplazar la población original y aplicar elitismo
    for(int i = 0; i < TAMANIO_AGE; ++i) {
      p_intermedia[i].fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, p_intermedia[i].w), tasaReduccion(p_intermedia[i].w));
      num_iteraciones++;
      nueva_p.insert(p_intermedia[i]);
    }

    // 6. Cambiar la población anterior (nueva generación)
    auto peor_cromosoma = p.begin();
    auto segundo_peor_cromosoma = ++p.begin();
    auto mejor_descendiente_actual = nueva_p.rbegin();
    auto segundo_mejor_descendiente_actual = ++nueva_p.rbegin();

    // NOTA: Este esquema de reemplazamiento solo es válido cuando TAMANIO_AGE == 2

    // Caso 1: Ambos descendientes sobreviven
    if(segundo_mejor_descendiente_actual->fitness > segundo_peor_cromosoma->fitness) {
      p.erase(segundo_peor_cromosoma);
      p.erase(p.begin());
      p.insert(*segundo_mejor_descendiente_actual);
      p.insert(*mejor_descendiente_actual);
    }
    // Caso 2: Solo el mejor descendiente sobrevive
    else if (mejor_descendiente_actual->fitness > peor_cromosoma->fitness) {
      p.erase(peor_cromosoma);
      p.insert(*mejor_descendiente_actual);
    }

    num_generaciones++;
  }

  // Elegir como solución el mejor cromosoma
  w = p.rbegin()->w;

  return w;
}



// *************************************************************************
// ********************** Algoritmos Meméticos (AMMs) **********************
// *************************************************************************

vector<double> AM_All(const vector<Ejemplo> entrenamiento) {

  Poblacion p;
  Poblacion::reverse_iterator mejor_padre;  // Elitismo

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;
  int num_generaciones = 1;
  int num_genes_individuo = w.size();
  int num_total_genes = w.size() * TAMANIO_AM;
  int num_cruces_esperados = PROBABILIDAD_CRUCE * (TAMANIO_AM / 2);  // Número de cruces esperados

  uniform_int_distribution<int> distribucion_uniform_int(0, num_total_genes - 1);

  // 1. Construir y evaluar una población inicial
  inicializarPoblacion(p, TAMANIO_AM, w.size(), entrenamiento);
  num_iteraciones += TAMANIO_AM;

  while(num_iteraciones < MAX_ITER) {

    PoblacionIntermedia p_intermedia;
    Poblacion nueva_p;

    mejor_padre = p.rbegin();  // Guardamos el mejor padre para elitismo

    // 2. Seleccionar población intermedia (ya evaluada)
    p_intermedia.resize(TAMANIO_AM);

    for(int i = 0; i < TAMANIO_AM; ++i)
      p_intermedia[i] = seleccion(p);

    // 3. Cruzar la población intermedia con el operador BLX
    for(int i = 0; i < 2 * num_cruces_esperados; i += 2) {
      auto descendientes = cruceBLX(p_intermedia[i], p_intermedia[i+1]);
      p_intermedia[i] = descendientes.first;
      p_intermedia[i+1] = descendientes.second;
    }

    // 4. Mutar la población intermedia
    set<int> mutados;
    int num_mutaciones_esperadas = mutacionesEsperadas(num_genes_individuo, num_total_genes);

    for(int i = 0; i < num_mutaciones_esperadas; ++i) {
      int comp;

      // Seleccionar componente a mutar, sin repetición
      while(mutados.size() == i) {
        comp = Random::get(distribucion_uniform_int);
        mutados.insert(comp);
      }

      int comp_seleccionada = comp / w.size();
      int gen_seleccionado = comp % w.size();

      mutacion(p_intermedia[comp_seleccionada], gen_seleccionado);
    }

    // 5. Evaluar, reemplazar la población original y aplicar elitismo
    for(int i = 0; i < TAMANIO_AM; ++i) {

      if(p_intermedia[i].fitness == -1.0) {
        p_intermedia[i].fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, p_intermedia[i].w), tasaReduccion(p_intermedia[i].w));
        num_iteraciones++;
      }

      nueva_p.insert(p_intermedia[i]);
    }

    auto mejor_padre_actual = nueva_p.rbegin();

    if(mejor_padre_actual->fitness < mejor_padre->fitness) {
      // Reemplazar el peor cromosoma de la población intermedia
      nueva_p.erase(nueva_p.begin());
      nueva_p.insert(*mejor_padre);
    }

    // 6. Reemplazar completamente la población previa 
    p = nueva_p;

    // 7. Cada 10 generaciones, aplicar la BL de baja intensidad sobre todos los cromosomas de la población
    if (num_generaciones % FRECUENCIA_BL_AM == 0) {

      nueva_p.clear();
      for(auto it = p.begin(); it != p.end(); ++it) {
        Cromosoma c = *it;
        num_iteraciones += busquedaLocalBajaIntensidad(entrenamiento, c);
        nueva_p.insert(c);
      }

      p = nueva_p;
    }

    // 8. Nueva generación
    num_generaciones++;
  }

  // Elegir como solución el mejor cromosoma
  w = p.rbegin()->w;

  return w;
}

vector<double> AM_Rand(const vector<Ejemplo> entrenamiento) {

  Poblacion p;
  Poblacion::reverse_iterator mejor_padre;  // Elitismo

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;
  int num_generaciones = 1;
  int num_genes_individuo = w.size();
  int num_total_genes = w.size() * TAMANIO_AM;
  int num_cruces_esperados = PROBABILIDAD_CRUCE * (TAMANIO_AM / 2);  // Número de cruces esperados

  uniform_int_distribution<int> distribucion_uniform_int(0, num_total_genes - 1);

  // 1. Construir y evaluar una población inicial
  inicializarPoblacion(p, TAMANIO_AM, w.size(), entrenamiento);
  num_iteraciones += TAMANIO_AM;

  while(num_iteraciones < MAX_ITER) {

    PoblacionIntermedia p_intermedia;
    Poblacion nueva_p;

    mejor_padre = p.rbegin();  // Guardamos el mejor padre para elitismo

    // 2. Seleccionar población intermedia (ya evaluada)
    p_intermedia.resize(TAMANIO_AM);

    for(int i = 0; i < TAMANIO_AM; ++i)
      p_intermedia[i] = seleccion(p);

    // 3. Cruzar la población intermedia con el operador BLX
    for(int i = 0; i < 2 * num_cruces_esperados; i += 2) {
      auto descendientes = cruceBLX(p_intermedia[i], p_intermedia[i+1]);
      p_intermedia[i] = descendientes.first;
      p_intermedia[i+1] = descendientes.second;
    }

    // 4. Mutar la población intermedia
    set<int> mutados;
    int num_mutaciones_esperadas = mutacionesEsperadas(num_genes_individuo, num_total_genes);

    for(int i = 0; i < num_mutaciones_esperadas; ++i) {
      int comp;

      // Seleccionar componente a mutar, sin repetición
      while(mutados.size() == i) {
        comp = Random::get(distribucion_uniform_int);
        mutados.insert(comp);
      }

      int comp_seleccionada = comp / w.size();
      int gen_seleccionado = comp % w.size();

      mutacion(p_intermedia[comp_seleccionada], gen_seleccionado);
    }

    // 5. Evaluar, reemplazar la población original y aplicar elitismo
    for(int i = 0; i < TAMANIO_AM; ++i) {

      if(p_intermedia[i].fitness == -1.0) {
        p_intermedia[i].fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, p_intermedia[i].w), tasaReduccion(p_intermedia[i].w));
        num_iteraciones++;
      }

      nueva_p.insert(p_intermedia[i]);
    }

    auto mejor_padre_actual = nueva_p.rbegin();

    if(mejor_padre_actual->fitness < mejor_padre->fitness) {
      // Reemplazar el peor cromosoma de la población intermedia
      nueva_p.erase(nueva_p.begin());
      nueva_p.insert(*mejor_padre);
    }

    // 6. Reemplazar completamente la población previa 
    p = nueva_p;

    // 7. Cada 10 generaciones, aplicar la BL de baja intensidad sobre un subconjunto de cromosomas de la población 
    //    seleccionado aleatoriamente con probabilidad 0.1 para cada cromosoma
    if (num_generaciones % FRECUENCIA_BL_AM == 0) {

      for(int i = 0; i < TAMANIO_AM * PROBABILIDAD_BUSQUEDA_LOCAL_AM; ++i) {
        uniform_int_distribution<int> distribucion2_uniform_int(0, TAMANIO_AM - 1);
        auto it = p.begin();
        advance(it, Random::get(distribucion2_uniform_int));
        Cromosoma c = *it;
        num_iteraciones += busquedaLocalBajaIntensidad(entrenamiento, c);
        p.erase(it);
        p.insert(c);
      }
    }

    // 8. Nueva generación
    num_generaciones++;
  }

  // Elegir como solución el mejor cromosoma
  w = p.rbegin()->w;

  return w;
}

vector<double> AM_Best(const vector<Ejemplo> entrenamiento) {

  Poblacion p;
  Poblacion::reverse_iterator mejor_padre;  // Elitismo

  vector<double> w(entrenamiento[0].num_caracts);

  int num_iteraciones = 0;
  int num_generaciones = 1;
  int num_genes_individuo = w.size();
  int num_total_genes = w.size() * TAMANIO_AM;
  int num_cruces_esperados = PROBABILIDAD_CRUCE * (TAMANIO_AM / 2);  // Número de cruces esperados

  uniform_int_distribution<int> distribucion_uniform_int(0, num_total_genes - 1);

  // 1. Construir y evaluar una población inicial
  inicializarPoblacion(p, TAMANIO_AM, w.size(), entrenamiento);
  num_iteraciones += TAMANIO_AM;

  while(num_iteraciones < MAX_ITER) {

    PoblacionIntermedia p_intermedia;
    Poblacion nueva_p;

    mejor_padre = p.rbegin();  // Guardamos el mejor padre para elitismo

    // 2. Seleccionar población intermedia (ya evaluada)
    p_intermedia.resize(TAMANIO_AM);

    for(int i = 0; i < TAMANIO_AM; ++i)
      p_intermedia[i] = seleccion(p);

    // 3. Cruzar la población intermedia con el operador BLX
    for(int i = 0; i < 2 * num_cruces_esperados; i += 2) {
      auto descendientes = cruceBLX(p_intermedia[i], p_intermedia[i+1]);
      p_intermedia[i] = descendientes.first;
      p_intermedia[i+1] = descendientes.second;
    }

    // 4. Mutar la población intermedia
    set<int> mutados;
    int num_mutaciones_esperadas = mutacionesEsperadas(num_genes_individuo, num_total_genes);

    for(int i = 0; i < num_mutaciones_esperadas; ++i) {
      int comp;

      // Seleccionar componente a mutar, sin repetición
      while(mutados.size() == i) {
        comp = Random::get(distribucion_uniform_int);
        mutados.insert(comp);
      }

      int comp_seleccionada = comp / w.size();
      int gen_seleccionado = comp % w.size();

      mutacion(p_intermedia[comp_seleccionada], gen_seleccionado);
    }

    // 5. Evaluar, reemplazar la población original y aplicar elitismo
    for(int i = 0; i < TAMANIO_AM; ++i) {

      if(p_intermedia[i].fitness == -1.0) {
        p_intermedia[i].fitness = fitness(tasaClasificacionLeaveOneOut(entrenamiento, p_intermedia[i].w), tasaReduccion(p_intermedia[i].w));
        num_iteraciones++;
      }

      nueva_p.insert(p_intermedia[i]);
    }

    auto mejor_padre_actual = nueva_p.rbegin();

    if(mejor_padre_actual->fitness < mejor_padre->fitness) {
      // Reemplazar el peor cromosoma de la población intermedia
      nueva_p.erase(nueva_p.begin());
      nueva_p.insert(*mejor_padre);
    }

    // 6. Reemplazar completamente la población previa 
    p = nueva_p;

    // 7. Cada 10 generaciones, aplica la BL sobre los 0.1*N mejores cromosomas de la población 
    //    actual (N es el tamaño de dicha población)
    if (num_generaciones % FRECUENCIA_BL_AM == 0) {

      for(int i = 0; i < TAMANIO_AM * PROBABILIDAD_BUSQUEDA_LOCAL_AM; ++i) {
        auto it = prev(p.end(), (i+1));  // Mejor cromosoma
        Cromosoma c = *it;
        num_iteraciones += busquedaLocalBajaIntensidad(entrenamiento, c);
        p.erase(it);
        p.insert(c);
      }
    }

    // 8. Nueva generación
    num_generaciones++;
  }

  // Elegir como solución el mejor cromosoma
  w = p.rbegin()->w;

  return w;
}



// *************************************************************************
// ************************ Resultados de Ejecución ************************
// *************************************************************************

void resultadosAGG_BLX() {

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
    cout << "************************************ " << nombre_archivo << " (AGG_BLX) ************************************************" << endl;

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

      w = AGG_BLX(entrenamiento);

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

void resultadosAGG_Arit() {

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
    cout << "************************************ " << nombre_archivo << " (AGG_Arit) ************************************************" << endl;

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

      w = AGG_Arit(entrenamiento);

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

void resultadosAGE_BLX() {

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
    cout << "************************************ " << nombre_archivo << " (AGE_BLX) ************************************************" << endl;

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

      w = AGE_BLX(entrenamiento);

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

void resultadosAGE_Arit() {

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
    cout << "************************************ " << nombre_archivo << " (AGE_Arit) ************************************************" << endl;

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

      w = AGE_Arit(entrenamiento);

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

void resultadosAM_All() {

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
    cout << "************************************ " << nombre_archivo << " (AM_All) ************************************************" << endl;

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

      w = AM_All(entrenamiento);

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

void resultadosAM_Rand() {

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
    cout << "************************************ " << nombre_archivo << " (AM_Rand) ************************************************" << endl;

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

      w = AM_Rand(entrenamiento);

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

void resultadosAM_Best() {

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
    cout << "************************************ " << nombre_archivo << " (AM_Best) ************************************************" << endl;

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

      w = AM_Best(entrenamiento);

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
