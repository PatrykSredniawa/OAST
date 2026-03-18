import numpy as np
import random

# --- KONFIGURACJA (Zgodnie z wymaganiami: MAX_GEN=10, Wynik=2) ---
N = 20          # Rozmiar populacji P(n)
K_ops = 10      # K operacji (generuje 20 potomków)
P_CH_MU = 0.3   # Prawdopodobieństwo mutacji chromosomu
P_GENE_MU = 0.3 # Prawdopodobieństwo mutacji genu
MAX_GEN = 10    # Limit generacji

LINKS_CAP = {1: 4, 2: 4, 3: 2, 4: 4, 5: 4}

DEMANDS = [
    {'id': 1, 'vol': 3, 'paths': [[1], [2, 3], [2, 4, 5]]},
    {'id': 2, 'vol': 4, 'paths': [[2], [1, 3], [1, 4, 5]]},
    {'id': 3, 'vol': 2, 'paths': [[1, 4], [2, 5]]},
    {'id': 4, 'vol': 2, 'paths': [[3], [1, 2], [4, 5]]},
    {'id': 5, 'vol': 3, 'paths': [[4], [3, 5], [1, 2, 5]]},
    {'id': 6, 'vol': 4, 'paths': [[5], [3, 4], [1, 2, 4]]}
]

class ChromosomeDAP:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = []
            for d in DEMANDS:
                alloc = np.zeros(len(d['paths']), dtype=int)
                for _ in range(d['vol']):
                    alloc[random.randint(0, len(d['paths'])-1)] += 1
                self.genes.append(alloc.tolist())
        else:
            self.genes = genes
        self.fitness, self.link_loads, self.overloads = self.evaluate()

    def evaluate(self):
        loads = {l: 0 for l in LINKS_CAP}
        for i, d in enumerate(DEMANDS):
            for p_idx, flow in enumerate(self.genes[i]):
                for link in d['paths'][p_idx]:
                    loads[link] += flow
        ov = {l: max(0, loads[l] - LINKS_CAP[l]) for l in LINKS_CAP}
        return sum(ov.values()), loads, ov

    def __lt__(self, other):
        return self.fitness < other.fitness

# --- DODATKOWE METODY DOBORU PAR (ROZSZERZENIE) ---
def select_pair(pop, mode='tournament'):
    if mode == 'random':
        return random.sample(pop, 2)
    elif mode == 'tournament':
        # Wybór najlepszego z losowej trójki (silna presja)
        def tour(): return min(random.sample(pop, 3))
        return tour(), tour()
    elif mode == 'elite':
        # Najlepszy + dowolny inny
        return pop[0], random.choice(pop[1:])

# --- OPERATORY ---
def crossover(p1, p2):
    c1_g, c2_g = [], []
    for g1, g2 in zip(p1.genes, p2.genes):
        if random.random() < 0.5:
            c1_g.append(g1[:]); c2_g.append(g2[:])
        else:
            c1_g.append(g2[:]); c2_g.append(g1[:])
    return ChromosomeDAP(c1_g), ChromosomeDAP(c2_g)

def mutate(chrom):
    if random.random() > P_CH_MU: return chrom
    new_genes = [g[:] for g in chrom.genes]
    for i, d in enumerate(DEMANDS):
        if random.random() < P_GENE_MU:
            flows = [idx for idx, v in enumerate(new_genes[i]) if v > 0]
            if flows:
                src, dst = random.choice(flows), random.randint(0, len(d['paths'])-1)
                new_genes[i][src] -= 1
                new_genes[i][dst] += 1
    return ChromosomeDAP(new_genes)

# --- IMPLEMENTACJA ZGODNA ZE SLAJDEM 17 ---
def run_ea_dap(mode='tournament'):
    # 1. n := 0; initialize(P(0)); sort(P(0))
    pop = sorted([ChromosomeDAP() for _ in range(N)])
    trajectory = [pop[0].fitness]

    # 2. while (warunek stopu) do
    for n in range(1, MAX_GEN + 1):
        # O := pusty zbiór (offspring pool)
        offspring_pool = []
        
        # 3. for i := 1 to K do O := O + crossover/mutate
        for _ in range(K_ops):
            p1, p2 = select_pair(pop, mode=mode)
            c1, c2 = crossover(p1, p2)
            offspring_pool.extend([mutate(c1), mutate(c2)])
        
        # 4. sort(O) - Użycie listy uporządkowanej dla potomstwa
        offspring_pool.sort()
        
        # 5. P(n) := select_best_N(P(n-1) + O) - Sukcesja elitarna
        combined = sorted(pop + offspring_pool)
        pop = combined[:N]
        
        trajectory.append(pop[0].fitness)
        print(f"Generacja {n}: Najlepszy fitness = {pop[0].fitness}")

    # RAPORT KOŃCOWY
    best = pop[0]
    print("\n" + "="*15 + " RAPORT KOŃCOWY DAP " + "="*15)
    print("LINK | LOAD | CAP | OVERLOAD") 
    for l in sorted(LINKS_CAP.keys()):
        print(f"{l:<4} | {best.link_loads[l]:<4} | {LINKS_CAP[l]:<3} | {best.overloads[l]}")
    print(f"\nWynik końcowy (Fitness): {best.fitness}")
    return best

if __name__ == "__main__":
    run_ea_dap(mode='tournament') 