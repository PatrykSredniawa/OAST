import random
import copy

# --- PARAMETRY (NIEPODLEGAJĄCE NEGOCJACJI) ---
N = 20          
K_ops = 10      
P_CH_MU = 0.1   
P_GENE_MU = 0.1 
MAX_GEN = 10    

# Konfiguracja rozszerzona dla lepszej zbieżności
SELECTED_SELECTION = 'tournament' 
SELECTED_MUTATION = 'swap'

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
                alloc = [0] * len(d['paths'])
                # Losowa inicjalizacja rozproszona
                for _ in range(d['vol']):
                    alloc[random.randint(0, len(d['paths']) - 1)] += 1
                self.genes.append(alloc)
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

def get_parent(pop):
    # Tournament Selection (k=2) - silniejsza presja selekcyjna
    c1, c2 = random.sample(pop, 2)
    return copy.deepcopy(c1 if c1.fitness <= c2.fitness else c2)

def mutate_op(chrom):
    if random.random() > P_CH_MU:
        return chrom
    new_genes = copy.deepcopy(chrom.genes)
    for i in range(len(DEMANDS)):
        if random.random() < P_GENE_MU:
            f = new_genes[i]
            if len(f) > 1:
                # SWAP: Zamiana wartości między dwiema różnymi ścieżkami
                idx1, idx2 = random.sample(range(len(f)), 2)
                f[idx1], f[idx2] = f[idx2], f[idx1]
    return ChromosomeDAP(new_genes)

def run_ea_final():
    # initialize(P(0))
    pop = sorted([ChromosomeDAP() for _ in range(N)])
    
    for gen in range(1, MAX_GEN + 1):
        O = []
        # K operacji krzyżowania -> 2K osobników
        for _ in range(K_ops):
            p1 = get_parent(pop)
            p2 = get_parent(pop)
            
            # Uniform Crossover na poziomie genów (popytów)
            c1_g, c2_g = [], []
            for g1, g2 in zip(p1.genes, p2.genes):
                if random.random() < 0.5:
                    c1_g.append(g1); c2_g.append(g2)
                else:
                    c1_g.append(g2); c2_g.append(g1)
            
            O.append(mutate_op(ChromosomeDAP(c1_g)))
            O.append(mutate_op(ChromosomeDAP(c2_g)))
        
        # P(n) := select_best_N[O U P(n-1)]
        combined = pop + O
        combined.sort()
        pop = combined[:N]
        
        print(f"Generacja {gen}: Najlepszy fitness = {pop[0].fitness}")

    best = pop[0]
    print("\n" + "="*15 + " RAPORT KOŃCOWY " + "="*15)
    print(f"KONFIGURACJA: Selekcja={SELECTED_SELECTION}, Mutacja={SELECTED_MUTATION}")
    print("-" * 46)
    print(f"{'LINK':<5} | {'LOAD':<5} | {'CAP':<5} | {'OVERLOAD':<10}")
    print("-" * 46)
    for l in sorted(LINKS_CAP.keys()):
        print(f"{l:<5} | {best.link_loads[l]:<5} | {LINKS_CAP[l]:<5} | {best.overloads[l]:<10}")
    print(f"\nOstateczny wynik funkcji celu: {best.fitness}")

if __name__ == "__main__":
    run_ea_final()