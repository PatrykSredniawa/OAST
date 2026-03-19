import numpy as np
import random
import math
import matplotlib.pyplot as plt

# --- DANE WEJŚCIOWE (bez zmian) ---
moduleCapacity = 1

links = {
    1: {"A": 1, "Z": 2, "modules": 4, "cost": 1},
    2: {"A": 1, "Z": 3, "modules": 4, "cost": 1},
    3: {"A": 2, "Z": 3, "modules": 2, "cost": 1},
    4: {"A": 2, "Z": 4, "modules": 4, "cost": 1},
    5: {"A": 3, "Z": 4, "modules": 4, "cost": 1},
}

demands = {
    1: {"A": 1, "Z": 2, "volume": 3, "paths": 3},
    2: {"A": 1, "Z": 3, "volume": 4, "paths": 3},
    3: {"A": 1, "Z": 4, "volume": 2, "paths": 2},
    4: {"A": 2, "Z": 3, "volume": 2, "paths": 3},
    5: {"A": 2, "Z": 4, "volume": 3, "paths": 3},
    6: {"A": 3, "Z": 4, "volume": 4, "paths": 3},
}

demand_paths = {
    (1, 1): [1], (1, 2): [2, 3], (1, 3): [2, 4, 5],
    (2, 1): [2], (2, 2): [1, 3], (2, 3): [1, 4, 5],
    (3, 1): [1, 4], (3, 2): [2, 5],
    (4, 1): [3], (4, 2): [1, 2], (4, 3): [4, 5],
    (5, 1): [4], (5, 2): [3, 5], (5, 3): [1, 2, 5],
    (6, 1): [5], (6, 2): [3, 4], (6, 3): [1, 2, 4],
}

# --- Parametry EA ---
N = 20
K = 10
p_mut = 0.1
q_gene = 0.1
GENERATIONS = 100
SEED = 0

# --- FUNKCJE POMOCNICZE (bez zmian) ---
def random_chromosome():
    chrom = {}
    for d, info in demands.items():
        P, h = info["paths"], info["volume"]
        flows = [0] * P
        for _ in range(h): flows[random.randrange(P)] += 1
        chrom[d] = flows
    return chrom

def repair(chrom):
    new = {}
    for d, info in demands.items():
        P, h = info["paths"], info["volume"]
        g = [max(0, int(round(x))) for x in chrom.get(d, [0]*P)[:P]]
        s = sum(g)
        if s == 0: g[0] = h
        else:
            scaled = [int(math.floor(x * h / s)) for x in g]
            diff = h - sum(scaled)
            for i in range(abs(diff)):
                idx = i % P
                if diff > 0: scaled[idx] += 1
                elif scaled[idx] > 0: scaled[idx] -= 1
            g = scaled
        new[d] = g
    return new

def link_loads_from_chrom(chrom):
    loads = {e: 0 for e in links}
    for d, flow_list in chrom.items():
        for p_idx, units in enumerate(flow_list, start=1):
            for e in demand_paths[(d, p_idx)]: loads[e] += units
    return loads

def objective_DAP(chrom):
    chrom = repair(chrom)
    loads = link_loads_from_chrom(chrom)
    O_pos = {e: max(0, load - links[e]["modules"] * moduleCapacity) for e, load in loads.items()}
    return max(O_pos.values()) if O_pos else 0, loads, O_pos

def objective_DDAP(chrom):
    chrom = repair(chrom)
    loads = link_loads_from_chrom(chrom)
    y = {e: (loads[e] + moduleCapacity - 1) // moduleCapacity for e in loads}
    cost = sum(links[e]["cost"] * y[e] for e in loads)
    return cost, loads, y

# --- OPERATORY (bez zmian) ---
def crossover_uniform(p1, p2):
    c1, c2 = {}, {}
    for d in demands:
        if random.random() < 0.5: c1[d], c2[d] = p1[d][:], p2[d][:]
        else: c1[d], c2[d] = p2[d][:], p1[d][:]
    return repair(c1), repair(c2)

def crossover_one_point(p1, p2):
    c1, c2 = {}, {}
    for d in demands:
        P = len(p1[d])
        pt = random.randint(1, P-1) if P > 1 else 1
        c1[d], c2[d] = p1[d][:pt] + p2[d][pt:], p2[d][:pt] + p1[d][pt:]
    return repair(c1), repair(c2)

def crossover_uniform_gene(p1, p2):
    c1, c2 = {}, {}
    for d in demands:
        c1[d], c2[d] = [], []
        for g1, g2 in zip(p1[d], p2[d]):
            if random.random() < 0.5: c1[d].append(g1); c2[d].append(g2)
            else: c1[d].append(g2); c2[d].append(g1)
    return repair(c1), repair(c2)

def mutate_shift(chrom):
    c = repair(chrom)
    for d, flows in c.items():
        if random.random() < q_gene and len(flows) > 1:
            nz = [idx for idx, v in enumerate(flows) if v > 0]
            if nz:
                # Rozbijamy na dwa kroki, aby uniknąć błędu zakresu (scope)
                i = random.choice(nz)
                # Teraz i jest już na pewno przypisane i widoczne dla list comprehension
                possible_j = [k for k in range(len(flows)) if k != i]
                j = random.choice(possible_j)
                
                flows[i] -= 1
                flows[j] += 1
    return c

# --- EA Z DODANYM ZBIERANIEM HISTORII ---
def run_EA(mode="DAP", generations=GENERATIONS, crossover_op=crossover_uniform, mutation_op=mutate_shift):
    if SEED is not None: random.seed(SEED)
    population = [random_chromosome() for _ in range(N)]
    history = [] # Lista do przechowywania najlepszego wyniku w każdej generacji

    def evaluate(ind):
        return objective_DAP(ind) if mode == "DAP" else objective_DDAP(ind)

    scored = [(evaluate(ch)[0], ch) for ch in population]
    scored.sort(key=lambda x: x[0])
    history.append(scored[0][0])

    for gen in range(1, generations + 1):
        offspring = []
        for _ in range(K):
            p1, p2 = random.sample(population, 2)
            c1, c2 = crossover_op(p1, p2)
            if random.random() < p_mut: c1 = mutation_op(c1)
            if random.random() < p_mut: c2 = mutation_op(c2)
            offspring.extend([c1, c2])

        off_scored = [(evaluate(ch)[0], ch) for ch in offspring]
        all_scored = sorted(scored + off_scored, key=lambda x: x[0])
        scored = all_scored[:N]
        population = [x[1] for x in scored]
        history.append(scored[0][0])

    return scored[0], history

def plot_results(all_histories):
    plt.figure(figsize=(12, 5))
    
    # Wykres dla DAP
    plt.subplot(1, 2, 1)
    for label, history in all_histories["DAP"]:
        # Dodajemy 'go-' dla zielonej linii z kropkami
        plt.plot(history, 'go-', label=label, markersize=4, markevery=10)
    plt.title("DAP Optimization Trajectory")
    plt.xlabel("Generation")
    plt.ylabel("Best chromosome fitness")
    plt.legend()
    plt.grid(True)

    # Wykres dla DDAP
    plt.subplot(1, 2, 2)
    for label, history in all_histories["DDAP"]:
        plt.plot(history, 'go-', label=label, markersize=4, markevery=10)
    plt.title("DDAP Optimization Trajectory")
    plt.xlabel("Generation")
    plt.ylabel("Best chromosome fitness")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    testy_krzyzowania = [
        ("Jednopunktowe", crossover_one_point),
        ("Równomierne", crossover_uniform),
        ("Równomierne (genowe)", crossover_uniform_gene),
    ]

    wyniki_tabele = []
    all_histories = {"DAP": [], "DDAP": []}

    for opis, crossover_op in testy_krzyzowania:
        # Test DAP
        best_dap, hist_dap = run_EA("DAP", crossover_op=crossover_op)
        wyniki_tabele.append(("DAP", opis, best_dap[0]))
        all_histories["DAP"].append((opis, hist_dap))

        # Test DDAP
        best_ddap, hist_ddap = run_EA("DDAP", crossover_op=crossover_op)
        wyniki_tabele.append(("DDAP", opis, best_ddap[0]))
        all_histories["DDAP"].append((opis, hist_ddap))

    # Tabela w konsoli
    print("\n" + "=" * 60)
    print(f"{'Tryb':<8} | {'Operator krzyżowania':<30} | {'F(x)':>5}")
    print("-" * 60)
    for tryb, opis, val in wyniki_tabele:
        print(f"{tryb:<8} | {opis:<30} | {val:>5}")
    print("=" * 60)

    # Generowanie wykresów
    plot_results(all_histories)