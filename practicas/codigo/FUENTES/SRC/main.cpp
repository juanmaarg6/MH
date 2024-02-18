// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica 3: Técnicas de Búsqueda basadas en Trayectorias
// Problema: APC

#include "p3.h"

using namespace std;

// *************************************************************************
// *************************** Función Principal ***************************
// *************************************************************************

int main(int argc, char * argv[]) {
  if (argc == 2) {
    long int semilla;

    semilla = stoi(argv[1]);

    Random::seed(semilla);

    //resultados1NN();
    //resultadosGreedy();
    //resultadosBusquedaLocal();
    //resultadosAGG_BLX();
    //resultadosAGG_Arit();
    //resultadosAGE_BLX();
    //resultadosAGE_Arit();
    //resultadosAM_All();
    //resultadosAM_Rand();
    //resultadosAM_Best();
    resultadosBMB();
    resultadosES();
    resultadosILS();
    resultadosILS_ES();
    resultadosVNS();
  }
  else
    cout << "ERROR: Falta la semilla de aleatoriedad. Para ejecutar: ./practica2_MH {semilla}" << endl;

  return 0;
}

