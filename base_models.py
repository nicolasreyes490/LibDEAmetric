
import pandas as pd
import math as mt
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs
#import highspy

EPSILON = 0.01
#name_data = ""
DMU = None

def init_data(name_data):

    global DMU
    DMU = pd.read_csv(name_data + ".csv")

    return None

def init_x(model):
    x = {}
    idx_input = 1
    for input in range(model.m + model.s + 1):
        if "input" in DMU.columns[input]:
            for j in model.J:
                x[idx_input, j] = DMU.iloc[j - 1, input]

            idx_input += 1

    return x


def init_y(model):
    y = {}
    idx_output = 1
    for output in range(model.m + model.s + 1):
        if "output" in DMU.columns[output]:
            for j in model.J:
                y[idx_output, j] = DMU.iloc[j - 1, output]

            idx_output += 1

    return y


def processing_x(model):
    x_ceil = {}

    for i in model.I:
        for j in model.J:
            min_value = mt.inf
            for tau in model.I:
                if model.x[tau, j] < min_value:
                    min_value = model.x[tau, j]

            if min_value > 0:
                x_ceil[i, j] = model.x[i, j]
            else:
                x_ceil[i, j] = model.x[i, j] - min_value + EPSILON

    return x_ceil


def processing_y(model):
    y_ceil = {}

    for r in model.R:
        for j in model.J:
            min_value = mt.inf
            for tau in model.R:
                if model.y[tau, j] < min_value:
                    min_value = model.y[tau, j]

            if min_value > 0:
                y_ceil[r, j] = model.y[r, j]
            else:
                y_ceil[r, j] = model.y[r, j] - min_value + EPSILON

    return y_ceil


def init_Mminus(model):
    Mminus = {}

    for i in model.I:
        max_value = 0
        for j in model.J:
            if model.x_ceil[i, j] > max_value:
                max_value = model.x_ceil[i, j]

        Mminus[i] = max_value

    return Mminus


def init_Mplus(model):
    Mplus = {}

    for r in model.R:
        max_value = 0
        for j in model.J:
            if model.y_ceil[r, j] > max_value:
                max_value = model.y_ceil[r, j]

        Mplus[r] = max_value

    return Mplus


def get_index(E, xi):
    for item in xi.items():
        if item[1] == E:
            return item[0]

    return None

def obj01_model(m, s, n, j0, SS):
    # create concrete model
    model = pyo.ConcreteModel()

    # parameters definition
    model.m = pyo.Param(within=pyo.NonNegativeIntegers, initialize=m)
    model.s = pyo.Param(within=pyo.NonNegativeIntegers, initialize=s)
    #model.n = pyo.Param(within=pyo.NonNegativeIntegers, initialize=n)

    if model.m > 0:
        model.I = pyo.RangeSet(1, model.m)
    if model.s > 0:
        model.R = pyo.RangeSet(1, model.s)
    model.J = pyo.Set(initialize=SS)

    if model.m > 0:
        model.x = pyo.Param(model.I, model.J, within=pyo.Reals, initialize=init_x)
    if model.s > 0:
        model.y = pyo.Param(model.R, model.J, within=pyo.Reals, initialize=init_y)

    #print("datos x:")
    #for j in model.J:
    #    for i in model.I:
    #        print(" ",model.x[i, j],end="")
    #    print("")

    if model.m > 0:
        model.x_ceil = pyo.Param(model.I, model.J, within=pyo.Reals, initialize=processing_x)
    if model.s > 0:
        model.y_ceil = pyo.Param(model.R, model.J, within=pyo.Reals, initialize=processing_y)

    if model.m > 0:
        model.Mminus = pyo.Param(model.I, within=pyo.Reals, initialize=init_Mminus)
    if model.s > 0:
        model.Mplus = pyo.Param(model.R, within=pyo.Reals, initialize=init_Mplus)

    # var decision definition
    if model.m > 0:
        model.sminus = pyo.Var(model.I,domain=pyo.Reals, initialize=0)
    if model.s > 0:
        model.splus = pyo.Var(model.R,domain=pyo.Reals, initialize=0)
    model.lamb = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=0)
    model.c = pyo.Var(domain=pyo.Binary,initialize=0)
    model.beta = pyo.Var(model.J, domain=pyo.Binary, initialize=0)

    model.psi = pyo.Var(model.J, model.J, domain=pyo.NonNegativeReals, initialize=0)

    # objective function definition
    if model.m > 0 and model.s > 0:
        model.of = pyo.Objective(rule=sum(abs(model.x[i, j] - model.x[i, k]) * model.psi[j, k] for i in model.I for j in model.J for k in model.J) + sum(abs(model.y[r, j] - model.y[r, k]) * model.psi[j, k] for r in model.R for j in model.J for k in model.J), sense=pyo.minimize)
    elif model.m > 0 and model.s == 0:
        model.of = pyo.Objective(rule=sum(abs(model.x[i, j] - model.x[i, k]) * model.psi[j, k] for i in model.I for j in model.J for k in model.J), sense=pyo.minimize)
    elif model.m == 0 and model.s > 0:
        model.of = pyo.Objective(rule=sum(abs(model.y[r, j] - model.y[r, k]) * model.psi[j, k] for r in model.R for j in model.J for k in model.J), sense=pyo.minimize)

    # constraint definition
    model.const = pyo.ConstraintList()

    # constraint (5)
    if model.m > 0:
        for i in model.I:
            model.const.add(model.x_ceil[i, j0] == sum(model.x_ceil[i, j] * model.lamb[j] for j in model.J if j != j0) + model.sminus[i])

    # constraint (6)
    if model.s > 0:
        for r in model.R:
            model.const.add(model.y_ceil[r, j0] == sum(model.y_ceil[r, j] * model.lamb[j] for j in model.J if j != j0) -model.splus[r])

    # constraint (7)
    model.const.add(sum(model.lamb[j] for j in model.J if j != j0) == 1)

    # constraint (8)
    if model.m > 0:
        for i in model.I:
            model.const.add(model.sminus[i] <= model.Mminus[i] * (1 - model.c))

    # constraint (9)
    if model.s > 0:
        for r in model.R:
            model.const.add(model.splus[r] <= model.Mplus[r] * (1 - model.c))

    # constraint (10)
    if model.m > 0:
        for i in model.I:
            model.const.add(model.sminus[i] >= (-1) * model.Mminus[i] * model.c)

    # constraint (11)
    if model.s > 0:
        for r in model.R:
            model.const.add(model.splus[r] >= (-1) * model.Mplus[r] * model.c)

    # constraint (12)
    model.const.add(sum(model.beta[j] for j in model.J if j != j0) <= model.m + model.s)

    # constraint (13)
    for j in model.J:
        model.const.add(model.lamb[j] <= model.beta[j])

    # constraint (23)
    for j in model.J:
        for k in model.J:
            model.const.add(model.psi[j, k] <= model.beta[j])

    # constraint (24)
    for j in model.J:
        for k in model.J:
            model.const.add(model.psi[j, k] <= model.beta[k])

    # constraint (25)
    for j in model.J:
        for k in model.J:
            model.const.add(model.psi[j, k] >= model.beta[j] + model.beta[k] - 1)

    # solver define
    solver = pyo.SolverFactory("glpk")

    # model is solve
    results = solver.solve(model, tee=False)
    #print(f"The solver returned a solution status of: {results.solver.termination_condition}"+" -- ",results.solver.status)
    #print(results)
    status_sol = results.solver.termination_condition
    # return info important
    of = None
    beta = {}
    c = 0

    # print(results)
    # print(solver.solve(model, tee = False))

    # if "optimal" in results["Solver"].termination_condition:
    of = model.of()
    for j in model.J:
        beta[j] = round(model.beta[j].value)

    c = round(model.c.value)
    # else:
    #    print(" Infeasible model 01 ....!")
    print(status_sol)
    return (of, beta, c, status_sol)

def intsbm_model(m, s, n, j0):

    #print(" Using INTSBM ...")

    # create concrete model
    model = pyo.ConcreteModel()

    # parameters definition
    model.m = pyo.Param(within=pyo.NonNegativeIntegers, initialize=m)
    model.s = pyo.Param(within=pyo.NonNegativeIntegers, initialize=s)
    model.n = pyo.Param(within=pyo.NonNegativeIntegers, initialize=n)
    model.I = pyo.RangeSet(1, model.m)
    if model.s > 0:
        model.R = pyo.RangeSet(1, model.s)
    model.J = pyo.RangeSet(1, model.n)
    model.x = pyo.Param(model.I, model.J, within=pyo.NonNegativeReals, initialize=init_x)
    if model.s > 0:
        model.y = pyo.Param(model.R, model.J, within=pyo.NonNegativeReals, initialize=init_y)
    model.j0 = pyo.Param(within=pyo.NonNegativeIntegers, initialize=j0)

    # decision variables definition
    model.lamb = pyo.Var(model.J, domain=pyo.Binary)
    model.gamma = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.zetaminus = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    if model.s > 0:
        model.zetaplus = pyo.Var(model.R, domain=pyo.NonNegativeReals)
    model.t = pyo.Var(domain=pyo.NonNegativeReals)

    # objective function definition
    model.of = pyo.Objective(rule=model.t - (1 / model.m) * sum(model.zetaminus[i] / model.x[i, model.j0] for i in model.I), sense=pyo.minimize)

    # constraint definition
    model.const = pyo.ConstraintList()

    # constraint (14)
    if model.s > 0:
        model.const.add(1 == model.t + (1 / model.s) * sum(model.zetaplus[r] / model.y[r, model.j0] for r in model.R))
    else:
        model.const.add(1 == model.t)

    # constraint (15)
    for i in model.I:
        model.const.add(model.x[i, model.j0] * model.t == sum(model.x[i, j] * model.gamma[j] for j in model.J) + model.zetaminus[i])

    # constraint (16)
    if model.s > 0:
        for r in model.R:
            model.const.add(model.y[r, model.j0] * model.t == sum(model.y[r, j] * model.gamma[j] for j in model.J) - model.zetaplus[r])

    # constraint (17)
    model.const.add(sum(model.gamma[j] for j in model.J) == model.t)

    # constraint (22)
    model.const.add(sum(model.lamb[j] for j in model.J) == 1)

    # constraint (23)
    for j in model.J:
        model.const.add(model.gamma[j] >= model.t - (1 - model.lamb[j]))

    # constraint (24)
    for j in model.J:
        model.const.add(model.gamma[j] <= model.t + (1 - model.lamb[j]))

    # solver define
    solver = pyo.SolverFactory("glpk")

    # model is solve
    results = solver.solve(model, tee=False)

    t = model.t.value

    #return t
    return model.of()


def obj02_model(m, s, n, j0, SS, beta, c):
    # t = instsbm_model(m, s, n, j0)    # OJO en la fo puede divir por cero

    # create concrete model 02
    model = pyo.ConcreteModel()

    # parameters definition
    model.m = pyo.Param(within=pyo.NonNegativeIntegers, initialize=m)
    model.s = pyo.Param(within=pyo.NonNegativeIntegers, initialize=s)
    #model.n = pyo.Param(within=pyo.NonNegativeIntegers, initialize=n)

    if model.m > 0:
        model.I = pyo.RangeSet(1, model.m)
    if model.s > 0:
        model.R = pyo.RangeSet(1, model.s)

    model.J = pyo.Set(initialize=SS)

    if model.m > 0:
        model.x = pyo.Param(model.I, model.J, within=pyo.Reals, initialize=init_x)
    if model.s > 0:
        model.y = pyo.Param(model.R, model.J, within=pyo.Reals, initialize=init_y)

    if model.m > 0:
        model.x_ceil = pyo.Param(model.I, model.J, within=pyo.Reals, initialize=processing_x)
    if model.s > 0:
        model.y_ceil = pyo.Param(model.R, model.J, within=pyo.Reals, initialize=processing_y)

    if model.m > 0:
        model.Mminus = pyo.Param(model.I, within=pyo.Reals, initialize=init_Mminus)
    if model.s > 0:
        model.Mplus = pyo.Param(model.R, within=pyo.Reals, initialize=init_Mplus)

    #for r in model.R:
    #    for j in model.J:
    #        print("", model.y_ceil[r, j], end="")
    #    print("")

    model.beta = pyo.Param(model.J, within=pyo.Binary, initialize=beta)
    model.c = pyo.Param(within=pyo.Binary, initialize=c)
    #for j in model.J:
    #    print(model.beta[j])
    #print(model.c())

    # definition variables decision
    model.fi = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
    model.lamb2 = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=0)
    if model.m > 0:
        model.sminus = pyo.Var(model.I, domain=pyo.Reals,initialize=0)
    if model.s > 0:
        model.splus = pyo.Var(model.R, domain=pyo.Reals, initialize=0)
    model.t = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
    model.C = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
    model.B = pyo.Var(model.J, domain=pyo.NonNegativeReals, initialize=0)

    # objective function definition
    model.of = pyo.Objective(rule=model.fi, sense=pyo.minimize)

    # constraint definition
    model.const = pyo.ConstraintList()

    # constraint (35)
    if model.m > 0:
        model.const.add(model.fi >= 1 - (model.t - (1 / model.m) * sum(model.sminus[i] / model.x_ceil[i, j0] for i in model.I)))
    else:
        model.const.add(model.fi >= 1 - model.t)

    # constraint (36)
    if model.m > 0:
        model.const.add(model.fi >= (model.t - (1 / model.m) * sum(model.sminus[i] / model.x_ceil[i, j0] for i in model.I)) - 1)
    else:
        model.const.add(model.fi >= model.t - 1)

    # constraint (37)
    if model.s > 0:
        model.const.add(model.t + (1 / model.s) * sum(model.splus[r] / model.y_ceil[r, j0] for r in model.R) == 1)
    else:
        model.const.add(model.t == 1)

    # constraint (38)
    if model.m > 0:
        for i in model.I:
            model.const.add(model.x_ceil[i, j0] * model.t == sum(model.x_ceil[i, j] * model.lamb2[j] for j in model.J if j != j0) + model.sminus[i])

    # constraint (39)
    if model.s > 0:
        for r in model.R:
            model.const.add(model.y_ceil[r, j0] * model.t == sum(model.y_ceil[r, j] * model.lamb2[j] for j in model.J if j != j0) - model.splus[r])

    # constraint (40)
    model.const.add(sum(model.lamb2[j] for j in model.J if j != j0) == model.t)

    # constraint (41)
    if model.m > 0:
        for i in model.I:
            model.const.add(model.sminus[i] <= model.Mminus[i] * (model.t - model.C))

    # constraint (42)
    if model.m > 0:
        for i in model.I:
            model.const.add(model.sminus[i] >= (-1) * model.Mminus[i] * model.C)

    # constraint (43)
    if model.s > 0:
        for r in model.R:
            model.const.add(model.splus[r] <= model.Mplus[r] * (model.t - model.C))

    # constraint (44)
    if model.s > 0:
        for r in model.R:
            model.const.add(model.splus[r] >= (-1) * model.Mplus[r] * model.C)

    # constraint (45)
    model.const.add(sum(model.B[j] for j in model.J) <= (model.m + model.s) * model.t)

    # constraint (46)
    for j in model.J:
        model.const.add(model.lamb2[j] <= model.B[j])

    # constraint (64)
    for j in model.J:

        if model.beta[j] == 1:
            model.const.add(model.B[j] == model.t)
        if model.beta[j] == 0:
            model.const.add(model.B[j] == 0)

    # constraint (65)
    if model.c == 1:
        model.const.add(model.C == model.t)
    if model.c == 0:
        model.const.add(model.C == 0)

    # solver define
    solver = pyo.SolverFactory("glpk")

    # model is solve
    results = solver.solve(model, tee=False)
    #print(results.solver.termination_condition)

    # return info important
    of = 0
    # if results["Solver"].termination_condition == "optimal":
    of = model.of()
    #print(of)
    # else:
    #    print(" Infeasible model 01 ....!")

    return of

def lexord_procedure(m, s, n, j0, SS):

    ES1 = obj01_model(m, s, n, j0, SS)

    #rint(ES1[3])

    #print(" Optimal solution (obj1):       ", ES1[0])

    ES2 = None

    if ES1[0] != None:
        ES2 = obj02_model(m, s, n, j0, SS, ES1[1], ES1[2])

        #print(" Optimal solution (obj2):       ", ES2)

    return ES2


def performance_metric(name_data, m, s, n):

    init_data(name_data)

    #data_file = open(name_data+"_log.txt","w")
    #data_file.write("|--------------------------------------------------------------------------------------------------|\n")

    metrics = {"DMU":[], "Sigma":[],"SS":[],"Comp":[]}

    # main algorithm
    # AQUI IDENTIFICAR SOLUCIONES EXTREMAS Y luego correr.
    SS = identify_feasible(m, s, n)
    sigma = 0.0

    J = [j for j in range(1, n + 1)]
    I = [i for i in range(1, n + 1)]
    iter = 0
    for j in set(J) - set(SS):
        #data_file.write(" Iteration ("+str(iter+1)+")\n")
        xi = {}
        lista = []
        for i in set(I) - set(SS):
            lista = [i] + SS
            lista.sort()
            #print(" List DNU:    ", lista)
            #data_file.write(" DMU List: "+str(lista)+"\n")
            #print(" i: ",i)
            xi[i] = lexord_procedure(m, s, n, i, lista)  #lexord_procedure(m, s, n, i, lista)

        #print(" XI:     ", xi)
        #data_file.write(" XI: "+str(xi)+"\n")

        E = max(xi.values())
        l = get_index(E, xi)

        SS.append(l)
        sigma = sigma + E

        # save results
        metrics["DMU"].append(l)
        metrics["Sigma"].append(sigma)
        metrics["SS"].append(str(SS))
        metrics["Comp"].append(xi)
        #print(" DMU: ",l)
        #print("SS: ",SS)
        #print(" Sigma:      ", sigma)
        #data_file.write(" Sigma: "+str(sigma)+"\n")
        #data_file.write("|--------------------------------------------------------------------------------------------------|\n")
        iter += 1
        # se puede obtener el minimo, pero es mejor hacer un funcion que entreegar key y valor.

        # lexord_procedure(m, s, n, i, SS)
    # print(SS)
    # print(sigma)
    #data_file.close()

    datos = pd.DataFrame(metrics)
    datos.to_csv("results_"+name_data+".csv", index=False)

    return None

def identify_feasible(m, s, n):

    SS = []
    solutions = list(range(1, n+1))
    for j in range(1, n+1):
        sol = obj01_model(m, s, n, j, solutions)
        #print(sol[3])
        if sol[3] == "infeasible":
            SS.append(j)

    return SS

# probar factibilidad
#SS = list(range(1, 26))

#for j in range(1,31):
#    S = obj01_model(3, 0, 30, j, SS)

#print(identify_feasible(3, 0, 25))
##SS = identify_feasible(3, 0, 25)
##print(obj01_model(3, 0, 25, 2, SS))
#print(obj01_model(3, 0, 25, SS))
#lexord_procedure(3,0, 25, 2, SS)
#lexord_procedure(3,0, 25, 3, SS)
#SS = [1, 3, 30]
#n = 3  # len(SS)#
#lexord_procedure(3, 0, n, 3, SS) #list(range(1, 31)))
# AQUI ES, LA LINEA DE ABAJO

#performance_metric(3, 0, 71)
#for j in range(1, 30+1):
#    print(intsbm_model(3,0,30,j))
#name = "all_log"
#m=3
#s=0
#n=235
#data_file = open(name+".txt","w")
#data_file.write(" Using INTSBM on all methods .....!\n")
#data_file.write("|--------------------------------------------------------------|\n")
#for j in range(n):
#    data_file.write(" j0 ("+str(j+1)+"): "+str(intsbm_model(m, s, n, j+1))+"\n")
#    #print(" j0 ("+str(j+1)+"): ",intsbm_model(3,0,30, j+1))

#data_file.write("|--------------------------------------------------------------|\n")
#data_file.close()

#print(identify_feasible(3,0,25))
#SS = identify_feasible(3,0,25)
#lexord_procedure(3,0,25,2,SS)