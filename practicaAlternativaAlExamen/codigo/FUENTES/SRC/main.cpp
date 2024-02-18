// Autor: Juan Manuel Rodríguez Gómez
// Asignatura: Metaheurísticas
// Práctica Alternativa al Examen: Metaheurística Leaders and Followers
// Problema: APC

#include "pAlternativa.h"

using namespace std;

// *************************************************************************
// *************************** Función Principal ***************************
// *************************************************************************

int main(int argc, char * argv[]) {
  if (argc == 2) {
    long int semilla;

    semilla = stoi(argv[1]);

    Random::seed(semilla);

    resultadosLeadersAndFollowers();
    resultadosLeadersAndFollowersBL();
    resultadosLeadersAndFollowersModificado();
  }
  else
    cout << "ERROR: Falta la semilla de aleatoriedad. Para ejecutar: ./practicaAlt_MH {semilla}" << endl;

  return 0;
}
