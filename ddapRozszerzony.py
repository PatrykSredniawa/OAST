import numpy as np
import random

# --- PARAMETRY KONFIGURACYJNE ---
N = 20          
K_ops = 10      
P_CH_MU = 0.1   
P_GENE_MU = 0.1 
MAX_GEN = 10    # Zmieniono na 10 zgodnie z prośbą

# --- DANE WEJŚCIOWE (DDAP) ---
MOD_CAP = 1     
MOD_COST = 1    
LINKS = [1, 2, 3, 4, 5]
DEMANDS = [
    {'id': 1, 'vol': 3, 'paths': [[1], [2, 3], [2, 4, 5]]},
    {'id': 2, 'vol': 4, 'paths': [[2], [1, 3], [1, 4, 5]]},
    {'id': 3, 'vol': 2, 'paths': [[1, 4], [2, 5]]},
    {'id': 4, 'vol': 2, 'paths': [[3], [1, 2], [4, 5]]},
    {'id': 5, 'vol': 3, 'paths': [[4], [3, 5], [1, 2, 5]]},
    {'id': 6, 'vol': 4, 'paths': [[5], [3, 4], [1, 2, 4]]}
]

class Chromosome:
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
        self.fitness, self.link_loads = self.evaluate()

    def evaluate(self):
        loads = {l: 0 for l in LINKS}
        for i, d in enumerate(DEMANDS):
            for p_idx, flow in enumerate(self.genes[i]):
                for link in d['paths'][p_idx]:
                    loads[link] += flow
        
        # Koszt całkowity (Suma obciążeń przy CAP=1 i COST=1)
        total_cost = sum(loads.values())
        return total_cost, loads

    def __lt__(self, other):
        return self.fitness < other.fitness

# --- METODY DOBORU PAR ---

def get_probability_dist(N_size):
    weights = [1.0 / (i + 1) for i in range(N_size)]
    total = sum(weights)
    return [w / total for w in weights]

def select_pair(pop, mode='elite'):
    N_size = len(pop)
    probs = get_probability_dist(N_size)
    if mode == 'random':
        return random.sample(pop, 2)
    elif mode == 'elite':
        x = pop[0]
        y = np.random.choice(pop, p=probs)
        return x, y
    return np.random.choice(pop, size=2, replace=False, p=probs).tolist()

# --- OPERATORY ---

def crossover(p1, p2):
    c1_genes, c2_genes = [], []
    for g1, g2 in zip(p1.genes, p2.genes):
        if random.random() < 0.5:
            c1_genes.append(g1[:]); c2_genes.append(g2[:])
        else:
            c1_genes.append(g2[:]); c2_genes.append(g1[:])
    return Chromosome(c1_genes), Chromosome(c2_genes)

def mutate(chrom):
    if random.random() > P_CH_MU: return chrom
    new_genes = [g[:] for g in chrom.genes]
    for i, d in enumerate(DEMANDS):
        if random.random() < P_GENE_MU and sum(new_genes[i]) > 0:
            idx_from = random.choice([idx for idx, v in enumerate(new_genes[i]) if v > 0])
            idx_to = random.randint(0, len(d['paths']) - 1)
            new_genes[i][idx_from] -= 1
            new_genes[i][idx_to] += 1
    return Chromosome(new_genes)

# --- ALGORYTM (N+K) ---

def run_ea(selection_mode='elite'):
    # Inicjalizacja P(0) - lista uporządkowana
    pop = sorted([Chromosome() for _ in range(N)])
    
    for n in range(1, MAX_GEN + 1):
        offspring_pool = []
        for _ in range(K_ops):
            p1, p2 = select_pair(pop, mode=selection_mode)
            c1, c2 = crossover(p1, p2)
            offspring_pool.extend([mutate(c1), mutate(c2)])
        
        # O := uporządkowana lista potomstwa
        offspring_pool.sort()
        
        # P(n) := select_best_N[O + P(n-1)]
        pop = sorted(pop + offspring_pool)[:N]
        print(f"Generacja {n}: Najlepszy koszt = {pop[0].fitness}")
        
    # --- WYNIK KOŃCOWY ---
    best = pop[0]
    print("\nLINK | LOAD | MODULES | COST")
    for l in LINKS:
        load = best.link_loads[l]
        modules = int(np.ceil(load / MOD_CAP))
        cost = modules * MOD_COST
        print(f"{l:<4} | {load:<4} | {modules:<7} | {cost}")
    
    print(f"\nCałkowity koszt sieci: {best.fitness}")
    return best

if __name__ == "__main__":
    run_ea(selection_mode='elite')