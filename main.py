# COMPARACIÓ DELS TRES MÈTODES I ESTIMACIÓ DEL TEMPS D'APLICACIÓ

# PART 1: comparem Euler explícit, Euler implícit i Crank-Nicolson amb la solució analítica.
# PART 2: utilitzem Crank-Nicolson amb la font de calor apagada a un temps t_on variable per estimar el temps d'aplicació segons els criteris del tractament.

# El model és unidimensional i homogeni: no representa explícitament la geometria dels catèters ni calcula el camp elèctric 
# (aquesta seria una possible millora tal com s'explica a l'informe).
# La font +1 de l'equació adimensional és uniforme a tot el domini, de manera que l'estudi de t_on és una estimació basada en aquesta idealització.

import numpy as np
import matplotlib.pyplot as plt


# FORMAT DE LES GRÀFIQUES

TAMANY_TICKS = 12
TAMANY_ETIQUETES = 14
TAMANY_TITOLS = 13
TAMANY_LLEGENDES = 10


# PARÀMETRES FÍSICS

Vef = 40/np.sqrt(2)               # potencial eficaç [V]
cv = 3698                         # capacitat calorífica a volum constant [J/(kg K)]
rho = 1081                        # densitat volumètrica de massa [kg/m^3]
kappa = 0.56                      # conductivitat tèrmica [W/(m K)]
sigma = 0.472                     # conductivitat elèctrica [S/m]
T_c = 36.5 + 273.15               # temperatura natural del cos humà [K]
L = 0.02                          # amplada de la làmina [m]
alpha = kappa/(cv*rho)            # difusivitat tèrmica [m^2/s]
Pext = sigma*(Vef/L)**2            # generació volumètrica de calor per efecte Joule [W/m^3]
# En el model, estem assumint E = Vef/L i, per tant, Pext = sigma*E^2, és una font de calor uniforme.
# Quan fem adimensional l'equació, aquesta font uniforme es converteix en el terme +1.


# PARÀMETRES ADIMENSIONALS

# Nota: en aquest programa quan escrivim T, ens referim a la temperatura adimensional absoluta
# i la condició inicial és T=beta. També podríem definir l'increment de temperatura com theta=T-beta,
# però aleshores la condició inicial equivalent seria theta=0.

tau_0 = L**2/alpha                # temps característic de difusió
T_0 = (Vef**2 * sigma)/kappa      # temperatura característica
beta = T_c/T_0


# DISCRETITZACIÓ ESPACIAL
N = 101
deltax = 1/(N-1)
x = np.arange(0, N) * deltax


# PARÀMETRES GENERALS DELS MÈTODES NUMÈRICS

tinicial = 0
tfinal_comparacio = 0.025       # temps final normalitzat per comparar els mètodes

tol = 1e-10                      # tolerància per decidir quan convergeixen les aproximacions numèriques
max_iter = 10000                 # nombre màxim d'iteracions de Gauss-Seidel



# SOLUCIÓ ANALÍTICA (amb uns 1000 termes)

def Tsol(beta, x, t, k):
    # Solució analítica de l'equació adimensional amb T=beta a l'instant inicial
    # i als dos extrems. La solució s'expressa com una sèrie de Fourier de termes senars.
    # k indica el nombre de termes que retenim per aproximar la sèrie infinita.
    return beta + (4/np.pi**3)*sum(
        ((1-np.exp(-(2*m+1)**2 * np.pi**2 * t))
         * np.sin((2*m+1)*np.pi*x))/(2*m+1)**3
        for m in range(k)
    )



# EULER EXPLÍCIT
# Si l'últim pas es passaria de tfinal, el tallem per acabar just a tfinal


def calcula_euler_explicit(beta, x, deltax, tinicial, tfinal, r):

    # Pas de temps segons el ràtio que hem triat. El paràmetre r és gamma = deltat/deltax^2.
    # Per a l'equació de la calor 1D, Euler explícit és estable si r <= 0.5.
    deltat = r * deltax**2
    N_local = len(x)

    # Posem tot l'array de temperatures de partida amb el valor inicial beta.
    # Aquí T és la temperatura adimensional absoluta, no és l'increment de temperatura T-beta.
    T = np.zeros(N_local) + beta

    # Fixem els extrems perquè no canviïn (condicions de contorn)
    T[0] = beta
    T[-1] = beta

    temps = tinicial

    # Llistes per anar guardant dades pas a pas.
    # camp desa un perfil espacial per a cada instant, temps_guardats desa els instants corresponents.
    camp = [T.copy()]
    temps_guardats = [temps]

    # Anem avançant el temps fins a arribar al límit
    while temps < tfinal - 1e-15:

        # Retallem l'últim salt per no passar-nos de llarg
        pas = min(deltat, tfinal - temps)

        T_nova = T.copy()

        # Calculem el canvi de temperatura punt a punt per dins.
        # Fórmula d'Euler explícit:
        # T_j^(i+1) = T_j^i + (pas/deltax^2)*(T_(j+1)^i - 2*T_j^i + T_(j-1)^i) + pas.
        # Només actualitzem els nodes interiors perquè els extrems es mantenen a beta.
        for j in range(1, N_local-1):
            T_nova[j] = T[j] + (pas/deltax**2) * (
                T[j+1] - 2*T[j] + T[j-1]
            ) + pas

        # Passem els valors nous al nostre array principal
        T = T_nova
        temps = temps + pas

        # Guardem com va evolucionant la temperatura. Fem .copy() per desar un perfil
        # independent i evitar que els valors canviïn en les iteracions posteriors.
        camp.append(T.copy())
        temps_guardats.append(temps)

    # Retornem arrays amb l'evolució de la temperatura (encara adimensional)
    # i els instants guardats. La conversió a graus Celsius la fem només per les gràfiques.
    return np.array(camp), np.array(temps_guardats)



# EULER IMPLÍCIT (amb Gauss-Seidel)

def calcula_euler_implicit(beta, x, deltax, tinicial, tfinal, gamma, tol, max_iter):

    deltat = gamma * deltax**2
    Nt = int(round((tfinal-tinicial)/deltat))
    t = tinicial + np.arange(0, Nt+1)*deltat
    N_local = len(x)

    # Un cop definits els vectors x,t, construïm una matriu de temperatures (instants x nodes)
    T = np.zeros((Nt+1, N_local))

    # Imposem les condicions de contorn als extrems
    T[:, 0] = beta
    T[:, -1] = beta

    # Condició inicial: tota la làmina comença a temperatura uniforme beta
    T[0, :] = beta

    """
    MÈTODE GAUSS-SEIDEL

    Com que ja sabem les temperatures als extrems, que són beta,
    amb el mètode de Gauss-Seidel NOMÉS calcularem els nodes interiors j = 1,..., N-2.

    Necessitem plantejar el sistema d'equacions de forma matricial tal que AX = B.
    En aquest cas, hi ha N-2 incògnites interiors. La matriu A és tridiagonal:
    té 1+2*gamma a la diagonal principal, -gamma a les diagonals superior i inferior,
    i zeros a la resta de posicions.

    En aquest programa no construïm A explícitament perquè podem aplicar directament
    la fórmula de recurrència de Gauss-Seidel als nodes interiors, per tant, no cal construir A.
    """

    for i in range(1, Nt+1):

        # De l'equació d'Euler implícit, els elements dels nodes interiors de B són
        # B_j = T_j^(i-1) + deltat. El tros 1:-1 selecciona només els nodes interiors.
        B = T[i-1, 1:-1] + deltat

        # Afegim les condicions de contorn conegudes als dos extrems.
        # El primer i l'últim node interior tenen un veí que és frontera i val beta;
        # aquests termes passen al costat dret del sistema i aporten gamma*beta.
        B[0] += gamma*beta
        B[-1] += gamma*beta

        # Construïm el vector X amb les temperatures interiors de l'instant anterior.
        # És l'ansatz inicial T_j^i(0) = T_j^(i-1); .copy() crea una còpia independent.
        X = T[i-1, 1:-1].copy()

        """
        Fórmula de recurrència de Gauss-Seidel:

        X_j^(k+1) = (B_j + gamma*X_(j-1)^(k+1) + gamma*X_(j+1)^(k))
                    / (1+2*gamma)

        Utilitzem el valor nou del veí esquerre i el valor antic del veí dret.
        """

        for it in range(max_iter):
            X_old = X.copy()

            for j in range(N_local-2):
                if j == 0:
                    # No hi ha cap veí esquerre desconegut: la frontera beta ja s'ha inclòs a B[0].
                    esquerra = 0
                else:
                    esquerra = gamma * X[j-1]  # X[j-1] és el valor nou d'aquesta iteració.

                if j == N_local-3:
                    # No hi ha cap veí dret desconegut: la frontera beta ja s'ha inclòs a B[-1].
                    dreta = 0
                else:
                    dreta = gamma * X_old[j+1]  # X_old[j+1] encara és el valor de la iteració anterior.

                X[j] = (B[j] + esquerra + dreta)/(1 + 2*gamma)

            # Comprovem la convergència després d'actualitzar tots els nodes
            if np.max(np.abs(X - X_old)) < tol:
                break
        else:
            print("Avís: Gauss-Seidel no convergeix al pas", i)

        # Guardem les temperatures del nou instant als nodes interiors
        T[i, 1:-1] = X

    return T, t



# CRANK-NICOLSON (amb Gauss-Seidel)
#
# El mètode de Crank-Nicolson té en compte tant la informació de l'instant anterior com del nou.
# Si t_on=None, la font de calor actua durant tota la simulació.
# Si s'indica t_on, la font actua fins a aquest instant i després s'apaga.
# eta = gamma/2 = deltat/(2*deltax^2) és el coeficient que apareix a la fórmula.


def calcula_crank_nicolson(beta, x, deltax, tinicial, tfinal, gamma,
                           tol, max_iter, t_on=None):

    deltat = gamma * deltax**2
    eta = gamma / 2

    Nt = int(round((tfinal-tinicial)/deltat))
    t = tinicial + np.arange(0, Nt+1)*deltat
    N_local = len(x)

    # La matriu T té una fila per instant temporal i una columna per node espacial.
    # T[i,j] representa la temperatura adimensional a l'instant i i a la posició j.
    T = np.zeros((Nt+1, N_local))

    # Condicions de contorn: els dos extrems es mantenen a beta durant tota la simulació.
    T[:, 0] = beta
    T[:, -1] = beta

    # Condició inicial: tota la làmina comença a temperatura uniforme beta.
    T[0, :] = beta

    for i in range(1, Nt+1):

        # La font només actua fins a t_on; si t_on=None, actua tot el temps.
        # Calculem la durada efectiva de l'interval [t[i-1], t[i]] durant la qual el senyal està actiu.
        # Si t_on cau dins del pas temporal, només comptem la fracció anterior a t_on.
        if t_on is None:
            font = deltat
        else:
            font = max(0.0, min(t[i], t_on) - t[i-1])

        # MÈTODE GAUSS-SEIDEL PER A CRANK-NICOLSON
        #
        # La discretització de Crank-Nicolson dona el sistema:
        # -eta*T_(j-1)^i + (1+2*eta)*T_j^i - eta*T_(j+1)^i
        #   = eta*T_(j-1)^(i-1) + (1-2*eta)*T_j^(i-1)
        #     + eta*T_(j+1)^(i-1) + font.
        #
        # La matriu A és tridiagonal: 1+2*eta a la diagonal principal i -eta
        # a les diagonals superior i inferior. Apliquem directament la recurrència, sense construir A de manera explícita.
        #
        # Construïm el vector B, que conté les temperatures conegudes de l'instant anterior.
        # 0:-2 selecciona els veïns esquerres; 1:-1, els nodes interiors;
        # 2: selecciona els veïns drets. Les tres porcions tenen longitud N-2.
        B = (eta*T[i-1, 0:-2]
             + (1-2*eta)*T[i-1, 1:-1]
             + eta*T[i-1, 2:]
             + font)

        # Afegim les condicions de contorn conegudes de l'instant nou.
        # El primer i l'últim node interior tenen un veí de frontera amb temperatura beta, aquests termes s'afegeixen al vector B amb eta*beta.
        B[0] += eta*beta
        B[-1] += eta*beta

        # Construïm X amb les temperatures interiors de l'instant anterior com a ansatz inicial.
        # .copy() crea una còpia independent.
        X = T[i-1, 1:-1].copy()

        """
        Fórmula de recurrència de Gauss-Seidel per a Crank-Nicolson:

        X_j^(k+1) = (B_j + eta*X_(j-1)^(k+1) + eta*X_(j+1)^(k)) / (1+2*eta)

        Utilitzem el valor nou del veí esquerre i el valor antic del veí dret.
        """

        for it in range(max_iter):
            X_old = X.copy()

            for j in range(N_local-2):
                if j == 0:
                    # No hi ha veí esquerre desconegut; la frontera ja s'ha afegit a B[0].
                    esquerra = 0
                else:
                    esquerra = eta * X[j-1]  # Valor nou del veí esquerre.

                if j == N_local-3:
                    # No hi ha veí dret desconegut; la frontera ja s'ha afegit a B[-1].
                    dreta = 0
                else:
                    dreta = eta * X_old[j+1]  # Valor antic del veí dret.

                X[j] = (B[j] + esquerra + dreta)/(1 + 2*eta)

            # La convergència es comprova després d'haver actualitzat tots els nodes interiors.
            if np.max(np.abs(X - X_old)) < tol:
                break
        else:
            print("Gauss-Seidel no convergeix al pas", i)

        T[i, 1:-1] = X

    return T, t



# FUNCIÓ PER CALCULAR L'ERROR D'UNA SIMULACIÓ RESPECTE L'ANALÍTICA
# Retorna la matriu d'errors en graus (error adimensional * T_0)


def calcula_error(T, t, x, beta, T_0):
    # T i T_analitica són adimensionals. La diferència de temperatures,
    # multiplicada per T_0, queda expressada en kelvin; numèricament és igual
    # a la diferència en graus Celsius.
    T_analitica = np.zeros((len(t), len(x)))
    for i in range(len(t)):
        T_analitica[i, :] = Tsol(beta, x, t[i], 1000)
    return np.abs(T - T_analitica) * T_0



# PART 1. COMPARACIÓ DELS TRES MÈTODES NUMÈRICS


# CÀLCULS 

# Euler explícit: r=0.49 i r=0.25 compleixen la condició d'estabilitat r <= 0.5.
# També calculem r=0.51 per mostrar el cas que supera el límit d'estabilitat.
T_exp049, t_exp049 = calcula_euler_explicit(
    beta, x, deltax, tinicial, tfinal_comparacio, 0.49
)
T_exp025, t_exp025 = calcula_euler_explicit(
    beta, x, deltax, tinicial, tfinal_comparacio, 0.25
)
T_exp051, t_exp051 = calcula_euler_explicit(
    beta, x, deltax, tinicial, tfinal_comparacio, 0.51
)

# Euler implícit
T_imp1, t_imp1 = calcula_euler_implicit(
    beta, x, deltax, tinicial, tfinal_comparacio, 1.0, tol, max_iter
)
T_imp05, t_imp05 = calcula_euler_implicit(
    beta, x, deltax, tinicial, tfinal_comparacio, 0.5, tol, max_iter
)

# Crank-Nicolson amb la font activa durant tota la simulació
T_cn1, t_cn1 = calcula_crank_nicolson(
    beta, x, deltax, tinicial, tfinal_comparacio, 1.0, tol, max_iter
)
T_cn05, t_cn05 = calcula_crank_nicolson(
    beta, x, deltax, tinicial, tfinal_comparacio, 0.5, tol, max_iter
)

# Errors respecte de la solució analítica.
# El cas explícit r=0.51 es representa per visualitzar la inestabilitat,
# però no s'inclou a la comparació quantitativa dels errors dels casos estables.
error_exp049 = calcula_error(T_exp049, t_exp049, x, beta, T_0)
error_exp025 = calcula_error(T_exp025, t_exp025, x, beta, T_0)
error_imp1 = calcula_error(T_imp1, t_imp1, x, beta, T_0)
error_imp05 = calcula_error(T_imp05, t_imp05, x, beta, T_0)
error_cn1 = calcula_error(T_cn1, t_cn1, x, beta, T_0)
error_cn05 = calcula_error(T_cn05, t_cn05, x, beta, T_0)

# Error màxim a cada instant (màxim de cada fila de la matriu d'errors).
# L'error global serà el màxim de tota la matriu: tots els instants i tots els nodes.
errmax_exp049 = np.max(error_exp049, axis=1)
errmax_exp025 = np.max(error_exp025, axis=1)
errmax_imp1 = np.max(error_imp1, axis=1)
errmax_imp05 = np.max(error_imp05, axis=1)
errmax_cn1 = np.max(error_cn1, axis=1)
errmax_cn05 = np.max(error_cn05, axis=1)


# VALORS PER A LES TAULES DE L'INFORME

print("\n=======================================")
print("PART 1: COMPARACIÓ DELS MÈTODES NUMÈRICS")
print("=======================================")

print("Euler explícit r = 0.49:")
print("  error global =", np.max(error_exp049), " error final =", errmax_exp049[-1],
      " temps del màxim =", t_exp049[np.argmax(errmax_exp049)])
print("Euler explícit r = 0.25:")
print("  error global =", np.max(error_exp025), " error final =", errmax_exp025[-1],
      " temps del màxim =", t_exp025[np.argmax(errmax_exp025)])
print("Euler implícit gamma = 1:")
print("  error global =", np.max(error_imp1), " error final =", errmax_imp1[-1],
      " temps del màxim =", t_imp1[np.argmax(errmax_imp1)])
print("Euler implícit gamma = 0.5:")
print("  error global =", np.max(error_imp05), " error final =", errmax_imp05[-1],
      " temps del màxim =", t_imp05[np.argmax(errmax_imp05)])
print("Crank-Nicolson gamma = 1:")
print("  error global =", np.max(error_cn1), " error final =", errmax_cn1[-1],
      " temps del màxim =", t_cn1[np.argmax(errmax_cn1)])
print("Crank-Nicolson gamma = 0.5:")
print("  error global =", np.max(error_cn05), " error final =", errmax_cn05[-1],
      " temps del màxim =", t_cn05[np.argmax(errmax_cn05)])


# Solució analítica al temps final en graus Celsius.
# La solució Tsol és adimensional, primer multipliquem per T_0 per obtenir kelvin
# i després restem 273.15 per expressar la temperatura en graus Celsius.
T_analitica_final = Tsol(beta, x, tfinal_comparacio, 1000)*T_0 - 273.15


# ---------------- FIGURA 1: TEMPERATURA A t = 0.025 ----------------
# Per passar a °C: T(°C) = T_adim*T_0 - 273.15

plt.figure(figsize=(11, 4))

# Panell (a): tots els casos estables i l'analítica
plt.subplot(1, 2, 1)
plt.plot(x, T_exp049[-1, :]*T_0 - 273.15, color="tab:green", linestyle="-", label="Euler explícit, $r=0.49$")
plt.plot(x, T_exp025[-1, :]*T_0 - 273.15, color="tab:green", linestyle="--", label="Euler explícit, $r=0.25$")
plt.plot(x, T_imp1[-1, :]*T_0 - 273.15, color="tab:blue", linestyle="-", label="Euler implícit, $\\gamma=1.0$")
plt.plot(x, T_imp05[-1, :]*T_0 - 273.15, color="tab:blue", linestyle="--", label="Euler implícit, $\\gamma=0.5$")
plt.plot(x, T_cn1[-1, :]*T_0 - 273.15, color="tab:orange", linestyle="-", label="Crank-Nicolson, $\\gamma=1.0$")
plt.plot(x, T_cn05[-1, :]*T_0 - 273.15, color="tab:orange", linestyle="--", label="Crank-Nicolson, $\\gamma=0.5$")
plt.plot(x, T_analitica_final, "k:", linewidth=2, label="Solució analítica")
plt.xlabel(r"Posició normalitzada $\tilde{x}$", fontsize=TAMANY_ETIQUETES)
plt.ylabel("Temperatura (°C)", fontsize=TAMANY_ETIQUETES)
plt.title("(a)", loc="left", fontsize=TAMANY_TITOLS)
plt.legend(fontsize=TAMANY_LLEGENDES, loc="lower center")
plt.tick_params(axis="both", direction="in", top=True, right=True,
                labelsize=TAMANY_TICKS)
plt.grid(False)

# Panell (b): Euler explícit amb r = 0.51 (inestable, és a dir, no convergirà)
plt.subplot(1, 2, 2)
plt.plot(x, T_exp051[-1, :]*T_0 - 273.15, color="tab:red", label="Euler explícit, $r=0.51$")
plt.plot(x, T_analitica_final, "k:", linewidth=2, label="Solució analítica")
plt.xlabel(r"Posició normalitzada $\tilde{x}$", fontsize=TAMANY_ETIQUETES)
plt.ylabel("Temperatura (°C)", fontsize=TAMANY_ETIQUETES)
plt.title("(b)", loc="left", fontsize=TAMANY_TITOLS)
plt.legend(fontsize=TAMANY_LLEGENDES)
plt.tick_params(axis="both", direction="in", top=True, right=True,
                labelsize=TAMANY_TICKS)
plt.grid(False)

plt.tight_layout()
plt.savefig("Tvsx_conjunta.png", dpi=300, bbox_inches="tight")
plt.show()


# FIGURA 2: ERRORS 

plt.figure(figsize=(11, 4))

# Panell (a): error absolut en funció de la posició al temps final
ax_error_pos = plt.subplot(1, 2, 1)
plt.plot(x, error_exp049[-1, :], color="tab:green", linestyle="-", label="Euler explícit, $r=0.49$")
plt.plot(x, error_exp025[-1, :], color="tab:green", linestyle="--", label="Euler explícit, $r=0.25$")
plt.plot(x, error_imp1[-1, :], color="tab:blue", linestyle="-", label="Euler implícit, $\\gamma=1.0$")
plt.plot(x, error_imp05[-1, :], color="tab:blue", linestyle="--", label="Euler implícit, $\\gamma=0.5$")
plt.plot(x, error_cn1[-1, :], color="tab:orange", linestyle="-", label="Crank-Nicolson, $\\gamma=1.0$")
plt.plot(x, error_cn05[-1, :], color="tab:orange", linestyle="--", label="Crank-Nicolson, $\\gamma=0.5$")
plt.xlabel(r"Posició normalitzada $\tilde{x}$", fontsize=TAMANY_ETIQUETES)
plt.ylabel("Error absolut (°C)", fontsize=TAMANY_ETIQUETES)
plt.title(r"(a) $\tilde{t}=0.025$", loc="left", fontsize=TAMANY_TITOLS)
plt.tick_params(axis="both", direction="in", top=True, right=True,
                labelsize=TAMANY_TICKS)
plt.grid(False)

# Panell (b): error màxim en funció del temps, amb l'eix y logarítmic.
# Comencem a l'índex 1 perquè a t = 0 l'error és 0 i el logaritme no existeix.
plt.subplot(1, 2, 2)
plt.plot(t_exp049[1:], errmax_exp049[1:], color="tab:green", linestyle="-", label="Euler explícit, $r=0.49$")
plt.plot(t_exp025[1:], errmax_exp025[1:], color="tab:green", linestyle="--", label="Euler explícit, $r=0.25$")
plt.plot(t_imp1[1:], errmax_imp1[1:], color="tab:blue", linestyle="-", label="Euler implícit, $\\gamma=1.0$")
plt.plot(t_imp05[1:], errmax_imp05[1:], color="tab:blue", linestyle="--", label="Euler implícit, $\\gamma=0.5$")
plt.plot(t_cn1[1:], errmax_cn1[1:], color="tab:orange", linestyle="-", label="Crank-Nicolson, $\\gamma=1.0$")
plt.plot(t_cn05[1:], errmax_cn05[1:], color="tab:orange", linestyle="--", label="Crank-Nicolson, $\\gamma=0.5$")
plt.yscale("log")
plt.xlabel(r"Temps normalitzat $\tilde{t}$", fontsize=TAMANY_ETIQUETES)
plt.ylabel("Error absolut màxim (°C)", fontsize=TAMANY_ETIQUETES)
plt.title("(b)", loc="left", fontsize=TAMANY_TITOLS)
plt.tick_params(axis="both", direction="in", top=True, right=True,
                which="both", labelsize=TAMANY_TICKS)
plt.grid(False)

# Llegenda comuna a sota de les dues gràfiques
handles, labels = ax_error_pos.get_legend_handles_labels()
plt.figlegend(handles, labels, loc="lower center", ncol=3,
              fontsize=TAMANY_LLEGENDES, frameon=False)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig("errors_conjunta.png", dpi=300, bbox_inches="tight")
plt.show()



# PART 2. ESTIMACIÓ DEL TEMPS D'APLICACIÓ t_on AMB CRANK-NICOLSON

# Aquesta part utilitza un temps final més llarg perquè, un cop apagat el senyal,
# el teixit tingui temps d'evolucionar i refredar-se.
# A diferència de la PART 1, aquí la font de calor deixa d'actuar a t_on;
# després d'aquest instant, la temperatura continua evolucionant per difusió tèrmica.

tfinal_optimitzacio = 0.035     # ~100 s, perquè doni temps a refredar-se
gamma_optimitzacio = 0.1       # dt més petit que als altres apartats per afinar t_on

T50 = (50 + 273.15)/T_0        # temperatura de referència inferior [adimensional]
T80 = (80 + 273.15)/T_0        # temperatura de referència superior [adimensional]

# teixit malalt: nodes 38 a 62 (x entre 0.38 i 0.62).
# Aquests 25 nodes representen la regió central de 0.5 cm dins la paret de 2 cm;
# si es consideren les vores de les cel·les, la regió s'estén aproximadament de 0.375 a 0.625.
# La resta de nodes interiors es considera teixit sa.
J_MALALT_INICI = 38
J_MALALT_FINAL = 62
N_NODES_MALALTS = J_MALALT_FINAL - J_MALALT_INICI + 1


# Valors de t_on que volem provar (adimensionals).
# El bucle farà una simulació completa per a cada temps d'aplicació candidat.
valors_t_on = np.arange(0.0198, 0.0208 + 1e-9, 0.00005)

t_on_valids = []
dosi_valids = []
t_on_no_valids = []
dosi_no_valids = []

millor_t_on = None
millor_dosi = -np.inf
millor_Tmax_sa = None
millor_Tmax = None

for t_on in valors_t_on:

    # Calculem la temperatura amb Crank-Nicolson i apaguem la font a t_on
    T_opt, t_opt = calcula_crank_nicolson(
        beta, x, deltax, tinicial, tfinal_optimitzacio,
        gamma_optimitzacio, tol, max_iter, t_on=t_on
    )
    deltat_opt = t_opt[1] - t_opt[0]

    # Inicialitzem la temperatura màxima del teixit sa i la màxima global amb beta,
    # que és la temperatura inicial i també la temperatura imposada a les fronteres.
    Tmax_sa = beta
    Tmax = beta

    # La dosi és una mesura acumulada: a cada pas temporal comptem la fracció
    # de nodes malalts dins de l'interval de temperatura [50, 80) °C i la multipliquem
    # pel temps físic del pas. Així s'obté una dosi expressada en segons equivalents.
    dosi = 0

    for i in range(1, len(t_opt)):
        tractats = 0

        for j in range(1, N-1):
            # Actualitzem la temperatura màxima assolida a qualsevol node interior.
            if T_opt[i, j] > Tmax:
                Tmax = T_opt[i, j]

            # Els nodes de fora de la zona malalta corresponen al teixit sa.
            if j < J_MALALT_INICI or j > J_MALALT_FINAL:
                if T_opt[i, j] > Tmax_sa:
                    Tmax_sa = T_opt[i, j]

            # Comptem els nodes malalts que es troben dins de l'interval de temperatura definit
            # Dins de la zona malalta, comptem els nodes que es troben entre 50 °C i 80 °C.
            elif T_opt[i, j] >= T50 and T_opt[i, j] < T80:
                tractats = tractats + 1

        # Fracció de teixit malalt tractat durant aquest pas, en segons.
        # tractats/N_NODES_MALALTS és la fracció de la zona malalta dins de l'interval terapèutic;
        # deltat_opt*tau_0 converteix el pas temporal adimensional a segons.
        dosi = dosi + (tractats/N_NODES_MALALTS) * deltat_opt * tau_0

    # Criteris de validesa definits per a aquest model:
    # - el teixit sa ha de quedar per sota de 50 °C;
    # - la temperatura màxima de tota la làmina ha de quedar per sota de 80 °C.
    # Entre els temps d'aplicació vàlids, seleccionem el que proporciona la dosi més gran.
    valid = Tmax_sa < T50 and Tmax < T80

    if valid:
        t_on_valids.append(t_on*tau_0)
        dosi_valids.append(dosi)

        if dosi > millor_dosi:
            millor_t_on = t_on
            millor_dosi = dosi
            millor_Tmax_sa = Tmax_sa
            millor_Tmax = Tmax
    else:
        t_on_no_valids.append(t_on*tau_0)
        dosi_no_valids.append(dosi)

    print("t_on =", round(t_on*tau_0, 2), "s  Tmax sa =",
          round(Tmax_sa*T_0 - 273.15, 3),
          "  Tmax =", round(Tmax*T_0 - 273.15, 3),
          "  dosi =", round(dosi, 3), " ", valid)


if millor_t_on is None:
    raise RuntimeError(
        "Cap valor de t_on compleix els criteris de validesa. "
        "Cal ampliar o modificar el rang de valors provats."
    )

print()
print("=======================================")
print("PART 2: ESTIMACIÓ DEL TEMPS D'APLICACIÓ")
print("=======================================")
print("t_on òptim:", round(millor_t_on, 5), "=", round(millor_t_on*tau_0, 1), "s")
print("Tmax teixit sa:", round(millor_Tmax_sa*T_0 - 273.15, 2), "°C")
print("Tmax global:", round(millor_Tmax*T_0 - 273.15, 2), "°C")
print("dosi:", round(millor_dosi, 2), "s")


# Tornem a calcular la solució amb el millor t_on per representar l'evolució
T_opt, t_opt = calcula_crank_nicolson(
    beta, x, deltax, tinicial, tfinal_optimitzacio,
    gamma_optimitzacio, tol, max_iter, t_on=millor_t_on
)
temps_s = t_opt*tau_0


# FIGURA 3: EVOLUCIÓ TÈRMICA I DOSI 

plt.figure(figsize=(5.5, 7))

plt.subplot(2, 1, 1)
plt.plot(temps_s, T_opt[:, 50]*T_0 - 273.15, color="tab:red",
         label=r"Centre malalt ($\tilde{x}=0.50$)")
plt.plot(temps_s, T_opt[:, 38]*T_0 - 273.15, color="tab:orange",
         label=r"Vora malalta ($\tilde{x}=0.38$)")
plt.plot(temps_s, T_opt[:, 37]*T_0 - 273.15, color="tab:blue",
         label=r"Vora sana ($\tilde{x}=0.37$)")
plt.axhline(50, color="k", linestyle=":", linewidth=1)
plt.axvline(millor_t_on*tau_0, color="gray", linestyle="--", linewidth=1)
plt.text(millor_t_on*tau_0 + 1, 38, "senyal\napagat", fontsize=8, color="gray")
plt.xlabel("Temps (s)")
plt.ylabel("Temperatura (°C)")
plt.title("(a)", loc="left")
plt.legend(fontsize=8, loc="upper left")
plt.tick_params(direction="in", top=True, right=True)
plt.grid(False)

plt.subplot(2, 1, 2)
plt.plot(t_on_valids, dosi_valids, "o-", color="tab:green", markersize=3,
         label="Vàlid (teixit sa < 50 °C)")
plt.plot(t_on_no_valids, dosi_no_valids, "o-", color="lightgray", markersize=3,
         label="No vàlid (teixit sa ≥ 50 °C)")
plt.axvline(millor_t_on*tau_0, color="gray", linestyle="--", linewidth=1)
plt.xlabel(r"Temps d'aplicació $t_{on}$ (s)")
plt.ylabel("Dosi de tractament (s)")
plt.title("(b)", loc="left")
plt.legend(fontsize=8, loc="upper left")
plt.tick_params(direction="in", top=True, right=True)
plt.grid(False)

plt.tight_layout()
plt.savefig("solucio_problema.png", dpi=300, bbox_inches="tight")
plt.show()
