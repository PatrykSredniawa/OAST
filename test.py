import random
import copy
import statistics

# --- PARAMETRY NIEPODLEGAJĄCE NEGOCJACJI ---
N = 20          
K_ops = 10      
P_CH_MU = 0.1   
P_GENE_MU = 0.1 
MAX_GEN = 10    
TEST_REPETITIONS = 10 # Ile razy powtórzyć każdy test dla wiarygodności

# --- DANE SIECI ---
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
            self.genes = [[random.randint(0, len(d['paths'])-1) for _ in range(d['vol'])] for d in DEMANDS]
            # Konwersja na format zliczony: [2, 1, 0]
            self.genes = []
            for d in DEMANDS:
                f = [0] * len(d['paths'])
                for _ in range(d['vol']): f[random.randint(0, len(f)-1)] += 1
                self.genes.append(f)
        else:
            self.genes = genes
        self.fitness = self.evaluate()

    def evaluate(self):
        loads = {l: 0 for l in LINKS}
        for i, d in enumerate(DEMANDS):
            for p_idx, flow in enumerate(self.genes[i]):
                for l_id in d['paths'][p_idx]: loads[l_id] += flow
        
        if self.mode == 'DAP':
            return sum(max(0, loads[l] - LINKS[l]['cap_dap']) for l in LINKS)
        else:
            return sum(((loads[l] + MODULE_CAPACITY - 1) // MODULE_CAPACITY) * LINKS[l]['cost_ddap'] for l in LINKS)

    def __lt__(self, other): return self.fitness < other.fitness

def run_ea(mode, sel_meth, mut_meth):
    pop = sorted([Chromosome(mode) for _ in range(N)])
    for _ in range(MAX_GEN):
        O = []
        for _ in range(K_ops):
            # Selekcja
            if sel_meth == 'tournament': p1, p2 = min(random.sample(pop, 2)), min(random.sample(pop, 2))
            elif sel_meth == 'ranking': 
                w = list(range(len(pop), 0, -1))
                p1, p2 = random.choices(pop, weights=w, k=2)
            else: p1, p2 = random.sample(pop, 2)
            
            # Krzyżowanie
            c1_g, c2_g = [], []
            for g1, g2 in zip(p1.genes, p2.genes):
                if random.random() < 0.5:
                    c1_g.append(copy.deepcopy(g1)); c2_g.append(copy.deepcopy(g2))
                else:
                    c1_g.append(copy.deepcopy(g2)); c2_g.append(copy.deepcopy(g1))
            
            # Mutacja i dodanie do O
            for g in [c1_g, c2_g]:
                if random.random() < P_CH_MU:
                    idx = random.randint(0, len(DEMANDS)-1)
                    if random.random() < P_GENE_MU:
                        f = g[idx]
                        if len(f) > 1:
                            i, j = random.sample(range(len(f)), 2)
                            if mut_meth == 'swap': f[i], f[j] = f[j], f[i]
                            elif f[i] > 0: f[i] -= 1; f[j] += 1
                O.append(Chromosome(mode, g))
        pop = sorted(pop + O)[:N]
    return pop[0].fitness

# --- TESTY ---
methods_sel = ['random', 'tournament', 'ranking']
methods_mut = ['shift', 'swap']
modes = ['DAP', 'DDAP']

print(f"Rozpoczynam benchmark ({TEST_REPETITIONS} powtórzeń na kombinację)...")

for mode in modes:
    print(f"\n--- WYNIKI DLA TRYBU: {mode} ---")
    print(f"{'SELEKCJA':<12} | {'MUTACJA':<8} | {'ŚR. FITNESS':<12} | {'NAJLEPSZY'}")
    print("-" * 50)
    
    for sel in methods_sel:
        for mut in methods_mut:
            results = [run_ea(mode, sel, mut) for _ in range(TEST_REPETITIONS)]
            avg = statistics.mean(results)
            best = min(results)
            print(f"{sel:<12} | {mut:<8} | {avg:<12.2f} | {best}")