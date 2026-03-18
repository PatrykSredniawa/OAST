import random
import copy

# --- PARAMETRY ---
N = 20          # Rozmiar populacji P(n)
K_ops = 10      # K operacji (generuje 10 par potomków, co daje 2K osobników)
P_CH_MU = 0.1   # Prawdopodobieństwo mutacji chromosomu
P_GENE_MU = 0.1 # Prawdopodobieństwo mutacji genu
MAX_GEN = 10    # Limit generacji

# Dane sieci (DAP)
links = {
    1: {'cap': 4}, 2: {'cap': 4}, 3: {'cap': 2},
    4: {'cap': 4}, 5: {'cap': 4}
}

demands = {
    1: {'vol': 3, 'paths': [[1], [2, 3], [2, 4, 5]]},
    2: {'vol': 4, 'paths': [[2], [1, 3], [1, 4, 5]]},
    3: {'vol': 2, 'paths': [[1, 4], [2, 5]]},
    4: {'vol': 2, 'paths': [[3], [1, 2], [4, 5]]},
    5: {'vol': 3, 'paths': [[4], [3, 5], [1, 2, 5]]},
    6: {'vol': 4, 'paths': [[5], [3, 4], [1, 2, 4]]}
}

def calculate_metrics(chromosome):
    link_loads = {l_id: 0 for l_id in links}
    for d_id, path_flows in chromosome.items():
        for path_idx, flow in enumerate(path_flows):
            for link_id in demands[d_id]['paths'][path_idx]:
                link_loads[link_id] += flow
    
    total_overload = 0
    overloads = {}
    for l_id, load in link_loads.items():
        ov = max(0, load - links[l_id]['cap'])
        overloads[l_id] = ov
        total_overload += ov
    return total_overload, link_loads, overloads

def generate_random_chromosome():
    chrom = {}
    for d_id, d_data in demands.items():
        flows = [0] * len(d_data['paths'])
        for _ in range(d_data['vol']):
            flows[random.randint(0, len(flows)-1)] += 1
        chrom[d_id] = flows
    return chrom

def crossover(p1, p2):
    # Losowy dobór par i wymiana genów (popytów)
    c1, c2 = {}, {}
    for d_id in demands:
        if random.random() < 0.5:
            c1[d_id], c2[d_id] = copy.deepcopy(p1[d_id]), copy.deepcopy(p2[d_id])
        else:
            c1[d_id], c2[d_id] = copy.deepcopy(p2[d_id]), copy.deepcopy(p1[d_id])
    return c1, c2

def mutate(chrom):
    # Mutacja chromosomu
    if random.random() < P_CH_MU:
        for d_id in demands:
            # Mutacja konkretnego genu (popytu)
            if random.random() < P_GENE_MU:
                f = chrom[d_id]
                if len(f) > 1:
                    idx_from = random.choice([i for i, val in enumerate(f) if val > 0])
                    idx_to = random.choice([i for i in range(len(f)) if i != idx_from])
                    f[idx_from] -= 1
                    f[idx_to] += 1
    return chrom

# --- Pętla Algorytmu (N+K) ---
# n := 0; initialize(P(0))
pop = []
for _ in range(N):
    c = generate_random_chromosome()
    pop.append({'gene': c, 'fit': calculate_metrics(c)[0]})

# P(n) musi być listą uporządkowaną
pop.sort(key=lambda x: x['fit'])

for gen in range(1, MAX_GEN + 1):
    O = []
    # for i := 1 to K do O := O U crossover
    for _ in range(K_ops):
        r1, r2 = random.sample(pop, 2)
        s1, s2 = crossover(r1['gene'], r2['gene'])
        O.extend([{'gene': s1}, {'gene': s2}])
    
    # for x in O do mutate(x)
    for child in O:
        child['gene'] = mutate(child['gene'])
        child['fit'] = calculate_metrics(child['gene'])[0]
    
    # P(n) := select_best_N[O U P(n-1)]
    combined = pop + O
    combined.sort(key=lambda x: x['fit'])
    pop = combined[:N]

# --- Generowanie Raportu ---
best = pop[0]
total_ov, final_loads, final_ovs = calculate_metrics(best['gene'])

print("=============== RAPORT KOŃCOWY ===============")
print(f"{'LINK':<5} | {'LOAD':<5} | {'CAP':<5} | {'OVERLOAD':<10}")
print("-" * 45)
for l_id in sorted(links.keys()):
    print(f"{l_id:<5} | {final_loads[l_id]:<5} | {links[l_id]['cap']:<5} | {final_ovs[l_id]:<10}")

print(f"\nOstateczny wynik funkcji celu: {total_ov}")