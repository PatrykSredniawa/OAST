import random
import copy

# --- PARAMETRY (STAŁE) ---
N = 20          # Rozmiar populacji P(n)
K_ops = 10      # K operacji (generuje 2K potomków)
P_CH_MU = 0.1   # Prawdopodobieństwo mutacji chromosomu
P_GENE_MU = 0.1 # Prawdopodobieństwo mutacji genu
MAX_GEN = 10    # Limit generacji

# --- NOWE DANE WEJŚCIOWE (DDAP) ---
MODULE_CAPACITY = 1
LINKS = {
    1: {'cost': 1},
    2: {'cost': 1},
    3: {'cost': 1},
    4: {'cost': 1},
    5: {'cost': 1}
}

DEMANDS = [
    {'id': 1, 'vol': 3, 'paths': [[1], [2, 3], [2, 4, 5]]},
    {'id': 2, 'vol': 4, 'paths': [[2], [1, 3], [1, 4, 5]]},
    {'id': 3, 'vol': 2, 'paths': [[1, 4], [2, 5]]},
    {'id': 4, 'vol': 2, 'paths': [[3], [1, 2], [4, 5]]},
    {'id': 5, 'vol': 3, 'paths': [[4], [3, 5], [1, 2, 5]]},
    {'id': 6, 'vol': 4, 'paths': [[5], [3, 4], [1, 2, 4]]}
]

class ChromosomeDDAP:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = []
            for d in DEMANDS:
                alloc = [0] * len(d['paths'])
                for _ in range(d['vol']):
                    alloc[random.randint(0, len(d['paths'])-1)] += 1
                self.genes.append(alloc)
        else:
            self.genes = genes
        self.fitness, self.link_loads, self.link_modules = self.evaluate()

    def evaluate(self):
        # 1. Oblicz obciążenie krawędzi (Load)
        loads = {l: 0 for l in LINKS}
        for i, d in enumerate(DEMANDS):
            for p_idx, flow in enumerate(self.genes[i]):
                for link in d['paths'][p_idx]:
                    loads[link] += flow
        
        # 2. Oblicz liczbę potrzebnych modułów i koszt (Fitness)
        total_cost = 0
        modules = {}
        for l_id, load in loads.items():
            # Liczba modułów = sufit(obciążenie / pojemność modułu)
            n_modules = (load + MODULE_CAPACITY - 1) // MODULE_CAPACITY
            modules[l_id] = n_modules
            total_cost += n_modules * LINKS[l_id]['cost']
            
        return total_cost, loads, modules

    def __lt__(self, other):
        return self.fitness < other.fitness

def crossover_op(P_n):
    p1, p2 = random.sample(P_n, 2)
    c1_g, c2_g = [], []
    for g1, g2 in zip(p1.genes, p2.genes):
        if random.random() < 0.5:
            c1_g.append(copy.deepcopy(g1))
            c2_g.append(copy.deepcopy(g2))
        else:
            c1_g.append(copy.deepcopy(g2))
            c2_g.append(copy.deepcopy(g1))
    return ChromosomeDDAP(c1_g), ChromosomeDDAP(c2_g)

def mutate_op(chrom):
    if random.random() > P_CH_MU:
        return chrom
    new_genes = copy.deepcopy(chrom.genes)
    for i, d in enumerate(DEMANDS):
        if random.random() < P_GENE_MU:
            f = new_genes[i]
            if len(f) > 1:
                sources = [idx for idx, v in enumerate(f) if v > 0]
                if sources:
                    src = random.choice(sources)
                    dst = random.choice([idx for idx in range(len(f)) if idx != src])
                    f[src] -= 1
                    f[dst] += 1
    return ChromosomeDDAP(new_genes)

def run_ea_ddap():
    # n := 0; initialize(P(0))
    pop = sorted([ChromosomeDDAP() for _ in range(N)])
    
    for n_gen in range(1, MAX_GEN + 1):
        O = []
        # Krzyżowanie -> 2K osobników
        for _ in range(K_ops):
            child1, child2 = crossover_op(pop)
            O.extend([child1, child2])
        
        # Mutacja
        O_mutated = [mutate_op(x) for x in O]
        
        # Sukcesja (N+K)
        combined = pop + O_mutated
        combined.sort()
        pop = combined[:N]
        
        print(f"Generacja {n_gen}: Najlepszy koszt (Fitness) = {pop[0].fitness}")

    # Raport Końcowy zgodny z wymogami DDAP
    best = pop[0]
    print("\n=============== RAPORT KOŃCOWY (DDAP) ===============")
    print("LINK | LOAD | MODULES | CAPACITY | COST")
    print("-" * 50)
    for l in sorted(LINKS.keys()):
        capacity = best.link_modules[l] * MODULE_CAPACITY
        cost = best.link_modules[l] * LINKS[l]['cost']
        print(f"{l:<4} | {best.link_loads[l]:<4} | {best.link_modules[l]:<7} | {capacity:<8} | {cost}")
    
    
    print("\nStruktura przepływów najlepszego rozwiązania:")
    for i, d in enumerate(DEMANDS):
        print(f"Popyt {d['id']}: {best.genes[i]}")
    print(f"\nOstateczny wynik funkcji celu (Koszt całkowity): {best.fitness}")
if __name__ == "__main__":
    run_ea_ddap()