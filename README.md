# DOCUMENTACIÓ D'USUARI DEL CODI utilitzat en la PRÀCTICA 1 - MÈTODES NUMÈRICS II 

# 1. Requisits previs

Cal instal·lar Python 3.13.18 o una versió compatible i les llibreries de NumPy i Matplotlib.

# 2. Execució

Per executar el programa, cal obrir una terminal al directori amb el fitxer principal i executar:

    python main.py

## 3. Funcionament

El codi resol numèricament la llei de Fourier amb generació de calor per efecte Joule mitjançant els mètodes numèrics Euler explícit, implícit i Crank-Nicolson, determinant l'evolució espaciotemporal de la temperatura en el teixit cardíac. També s'avalua el seu error numèric absolut respecte de la solució analítica.

## 4. Resultats

Quan s'executa el codi, s'obtenen les gràfiques de la temperatura en funció de la posició normalitzada per tots els casos dels tres mètodes analitzats. Pels casos convergents, també s'hi representa l'error absolut en funció de la posició normalitzada fixat el temps final de 0.025, i l'error absolut màxim en funció del temps normalitzat. A més, s'imprimeix per pantalla el temps òptim per aplicar el senyal altern calculat pel mètode de Crank-Nicolson, així com la temperatura màxima del teixit sa, la temperatura màxima global i el temps de dosi. S'ha representat de manera addicional la temperatura en funció del temps i la dosi de tractament en funció del temps d'aplicació. 
