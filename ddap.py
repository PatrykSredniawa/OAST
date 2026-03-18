import numpy as np
import random

# --- KONFIGURACJA ZGODNA Z UWAGĄ 2 ---
N = 20          
K_ops = 10      
P_CH_MU = 0.1   
P_GENE_MU = 0.1 
MAX_GEN = 10   

# --- DANE WEJŚCIOWE DDAP ---
MOD_CAP = 1     # param moduleCapacity := 1
MOD_COST = 1    # param link_moduleCost := 1

DEMANDS = [
    {'id': 1, 'vol': 3, 'paths': [[1], [2, 3], [2, 4, 5]]},
    {'id': 2, 'vol': 4, 'paths': [[2], [1, 3], [1, 4, 5]]},
    {'id': 3, 'vol': 2, 'paths': [[1, 4], [2, 5]]},
    {'id': 4, 'vol': 2, 'paths': [[3], [1, 2], [4, 5]]},
    {'id': 5, 'vol': 3, 'paths': [[4], [3, 5], [1, 2, 5]]},
    {'id': 6, 'vol': 4, 'paths': [[5], [3, 4], [1, 2, 4]]}
]

LINKS = [1, 2, 3, 4, 5]

class ChromosomeDDAP:
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
        self.fitness, self.link_loads, self.link_costs = self.evaluate()

    def evaluate(self):
        loads = {l: 0 for l in LINKS}
        for i, d in enumerate(DEMANDS):
            for p_idx, flow in enumerate(self.genes[i]):
                for link in d['paths'][p_idx]:
                    loads[link] += flow
        
        # Koszt linku = liczba potrzebnych modułów * koszt modułu
        # modules = ceil(load / MOD_CAP)
        costs = {}
        for l in LINKS:
            modules = int(np.ceil(loads[l] / MOD_CAP))
            costs[l] = modules * MOD_COST
            
        # F celu: Suma kosztów wszystkich linków (DDAP)
        f_val = sum(costs.values())
        return f_val, loads, costs

    def __lt__(self, other):
        return self.fitness < other.fitness

def crossover(p1, p2):
    c1_genes, c2_genes = [], []
    for g1, g2 in zip(p1.genes, p2.genes):
        if random.random() < 0.5:
            c1_genes.append(g1[:]); c2_genes.append(g2[:])
        else:
            c1_genes.append(g2[:]); c2_genes.append(g1[:])
    return ChromosomeDDAP(c1_genes), ChromosomeDDAP(c2_genes)

def mutate(chrom):
    if random.random() > P_CH_MU:
        return chrom
    new_genes = [g[:] for g in chrom.genes]
    for i, d in enumerate(DEMANDS):
        if random.random() < P_GENE_MU:
            has_flow = [idx for idx, val in enumerate(new_genes[i]) if val > 0]
            if has_flow:
                src = random.choice(has_flow)
                dst = random.randint(0, len(d['paths']) - 1)
                new_genes[i][src] -= 1
                new_genes[i][dst] += 1
    return ChromosomeDDAP(new_genes)

def run_ea_ddap():
    pop = sorted([ChromosomeDDAP() for _ in range(N)])
    trajectory = [pop[0].fitness]
    
    for n in range(1, MAX_GEN + 1):
        offspring = []
        for _ in range(K_ops):
            p1, p2 = random.sample(pop, 2)
            c1, c2 = crossover(p1, p2)
            offspring.extend([mutate(c1), mutate(c2)])
        
        pop = sorted(pop + offspring)[:N]
        trajectory.append(pop[0].fitness)

    best = pop[0]
    print("\n" + "="*20 + " RAPORT KOŃCOWY DDAP " + "="*20)
    print(f"Najlepszy koszt (F celu): {best.fitness}")
    
    print("\nPRZEPŁYWY ŚCIEŻKOWE:")
    for i, d in enumerate(DEMANDS):
        print(f"Demand {d['id']}: {best.genes[i]}")
    
    print("\nLINK | LOAD | MODULES | COST")
    for l in LINKS:
        load = best.link_loads[l]
        modules = int(np.ceil(load / MOD_CAP))
        cost = best.link_costs[l]
        print(f"{l:<4} | {load:<4} | {modules:<7} | {cost}")
    
    print(f"\nCałkowity koszt sieci: {best.fitness}")

if __name__ == "__main__":
    run_ea_ddap()