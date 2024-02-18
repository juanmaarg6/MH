// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
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
// ********************* Inicializar Vector de Pesos ***********************
// *************************************************************************
void inicializarVectorPesos(vector<double>& w) {
  
  for(int i = 0; i < w.size(); ++i)
    w[i] = 0.0;
}