// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica Alternativa al Examen: Metaheurística Leaders and Followers
// Problema: APC

#include "util.h"

using namespace std;

// *************************************************************************
// ******************** Lectura y Normalización de Datos *******************
// *************************************************************************

vector<Ejemplo> leerFicheroARFF(string nombre_archivo) {

  vector<Ejemplo> ejemplos;

  ifstream archivo(nombre_archivo);
  string linea;
  bool leyendo_datos = false;

  while( getline(archivo, linea) ) {
    if(!leyendo_datos) {
      if (linea.find("@data") != string::npos)
        leyendo_datos = true;
      continue;
    }

    stringstream ss(linea);
    vector<string> valores;
    string valor;

    while( getline(ss, valor, ',') ) {
      valores.push_back(valor);
    }

    string clase = valores.back();
    valores.pop_back();
    vector<double> valores_atributos;
    for (auto& v : valores)
      valores_atributos.push_back(stod(v));

    Ejemplo e;
    e.val_caracts = valores_atributos;
    e.categoria = clase;
    e.num_caracts = valores_atributos.size();
    ejemplos.push_back(e);
  }
  
  return ejemplos;
}

void normalizarValores(vector<vector<Ejemplo>>& datos) {

  double valor_min;
  double valor_max;
  double valor;
  
  for(int i = 0; i < datos[0][0].num_caracts; ++i) {
    valor_min = datos[0][0].val_caracts[i];
    valor_max = datos[0][0].val_caracts[i];

    for(int j = 0; j < datos.size(); ++j)
      for(int k = 0; k < datos[j].size(); ++k) {
        valor = datos[j][k].val_caracts[i];

        if(valor > valor_max)
          valor_max = valor;
        if(valor < valor_min)
          valor_min = valor;
      }
      
    for(int j = 0; j < datos.size(); ++j)
      for(int k = 0; k < datos[j].size(); ++k)
        datos[j][k].val_caracts[i] = (datos[j][k].val_caracts[i]-valor_min) / (valor_max-valor_min);
  }
}



// *************************************************************************
// ************************** Funciones Distancia **************************
// *************************************************************************

double distanciaEuclidea(const Ejemplo& e1, const Ejemplo& e2) {

  double dist = 0.0;

  for(int i = 0; i < e1.num_caracts; ++i)
    dist += (e1.val_caracts[i] - e2.val_caracts[i]) * (e1.val_caracts[i] - e2.val_caracts[i]);
    
  return dist;
}

double distanciaEuclideaPesos(const Ejemplo& e1, const Ejemplo& e2, const vector<double>& w) {
  
  double dist = 0.0;

  for(int i = 0; i < w.size(); ++i)
    if(w[i] > 0.1)
      dist += w[i] * (e1.val_caracts[i] - e2.val_caracts[i]) * (e1.val_caracts[i] - e2.val_caracts[i]);
    
  return dist;
}



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
// ********************* Inicializar Vector de Pesos ***********************
// *************************************************************************
void inicializarVectorPesos(vector<double>& w) {
  
  for(int i = 0; i < w.size(); ++i)
    w[i] = 0.0;
}