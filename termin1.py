import random
import copy

# --- PARAMETRY ---
N = 20          # Rozmiar populacji P(n)
K_ops = 10      # K operacji (generuje 2K potomków)
P_CH_MU = 0.1   # Prawdopodobieństwo mutacji chromosomu
P_GENE_MU = 0.1 # Prawdopodobieństwo mutacji genu
MAX_GEN = 10    # Limit generacji

# --- DANE WEJŚCIOWE ---
MODULE_CAPACITY = 1
LINKS = {
    1: {'cap_dap': 4, 'cost_ddap': 1},
    2: {'cap_dap': 4, 'cost_ddap': 1},
    3: {'cap_dap': 2, 'cost_ddap': 1},
    4: {'cap_dap': 4, 'cost_ddap': 1},
    5: {'cap_dap': 4, 'cost_ddap': 1}
}

DEMANDS = [
    {'id': 1, 'vol': 3, 'paths': [[1], [2, 3], [2, 4, 5]]},
    {'id': 2, 'vol': 4, 'paths': [[2], [1, 3], [1, 4, 5]]},
    {'id': 3, 'vol': 2, 'paths': [[1, 4], [2, 5]]},
    {'id': 4, 'vol': 2, 'paths': [[3], [1, 2], [4, 5]]},
    {'id': 5, 'vol': 3, 'paths': [[4], [3, 5], [1, 2, 5]]},
    {'id': 6, 'vol': 4, 'paths': [[5], [3, 4], [1, 2, 4]]}
]

class Chromosome:
    def __init__(self, mode, genes=None):
        self.mode = mode
        if genes is None:
            self.genes = []
            for d in DEMANDS:
                alloc = [0] * len(d['paths'])
                for _ in range(d['vol']):
                    alloc[random.randint(0, len(d['paths'])-1)] += 1
                self.genes.append(alloc)
        else:
            self.genes = genes
        self.fitness, self.link_loads, self.extra = self.evaluate()

    def evaluate(self):
        loads = {l: 0 for l in LINKS}
        for i, d in enumerate(DEMANDS):
            for p_idx, flow in enumerate(self.genes[i]):
                for link in d['paths'][p_idx]:
                    loads[link] += flow
        
        if self.mode == 'DAP':
            overloads = {l: max(0, loads[l] - LINKS[l]['cap_dap']) for l in LINKS}
            return sum(overloads.values()), loads, overloads
        else: # DDAP
            modules = {l: (loads[l] + MODULE_CAPACITY - 1) // MODULE_CAPACITY for l in LINKS}
            cost = sum(modules[l] * LINKS[l]['cost_ddap'] for l in LINKS)
            return cost, loads, modules

    def __lt__(self, other):
        return self.fitness < other.fitness

def crossover_op(pop, mode):
    p1, p2 = random.sample(pop, 2)
    c1_g, c2_g = [], []
    for g1, g2 in zip(p1.genes, p2.genes):
        if random.random() < 0.5:
            c1_g.append(copy.deepcopy(g1)); c2_g.append(copy.deepcopy(g2))
        else:
            c1_g.append(copy.deepcopy(g2)); c2_g.append(copy.deepcopy(g1))
    return Chromosome(mode, c1_g), Chromosome(mode, c2_g)

def mutate_op(chrom):
    if random.random() > P_CH_MU: return chrom
    new_genes = copy.deepcopy(chrom.genes)
    for i in range(len(DEMANDS)):
        if random.random() < P_GENE_MU:
            f = new_genes[i]
            if len(f) > 1 and sum(f) > 0:
                src = random.choice([idx for idx, v in enumerate(f) if v > 0])
                dst = random.choice([idx for idx in range(len(f)) if idx != src])
                f[src] -= 1; f[dst] += 1
    return Chromosome(chrom.mode, new_genes)

def run_solver(mode):
    print(f"\n{'='*20} URUCHAMIAM: {mode} {'='*20}")
    pop = sorted([Chromosome(mode) for _ in range(N)])
    
    for n in range(1, MAX_GEN + 1):
        O = []
        for _ in range(K_ops):
            c1, c2 = crossover_op(pop, mode)
            O.extend([mutate_op(c1), mutate_op(c2)])
        
        combined = sorted(pop + O)
        pop = combined[:N]
        print(f"Gen {n}: Best Fitness = {pop[0].fitness}")

    best = pop[0]
    print(f"\n--- RAPORT KOŃCOWY {mode} ---")
    if mode == 'DAP':
        print("LINK | LOAD | CAP | OVERLOAD")
        for l in sorted(LINKS.keys()):
            print(f"{l:<4} | {best.link_loads[l]:<4} | {LINKS[l]['cap_dap']:<3} | {best.extra[l]}")
    else:
        print("LINK | LOAD | MODS | COST")
        for l in sorted(LINKS.keys()):
            print(f"{l:<4} | {best.link_loads[l]:<4} | {best.extra[l]:<4} | {best.extra[l]*LINKS[l]['cost_ddap']}")
    print(f"Ostateczny wynik funkcji celu: {best.fitness}")

if __name__ == "__main__":
    run_solver('DAP')
    run_solver('DDAP')