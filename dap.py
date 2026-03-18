import numpy as np
import random

N = 20          # Rozmiar populacji (N) 
K_ops = 10      # Liczba operacji (K), generuje 20 potomków 
P_CH_MU = 0.1   # p - prawdopodobieństwo mutacji chromosomu 
P_GENE_MU = 0.1 # q - prawdopodobieństwo mutacji genu 
MAX_GEN = 10  # Zwiększone dla stabilności przy losowej selekcji 

# --- DANE WEJŚCIOWE (Zgodnie z plikiem AMPL) ---
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
        
        # Obliczanie przeciążeń 
        ov = {l: max(0, loads[l] - LINKS_CAP[l]) for l in LINKS_CAP}
        # F celu: suma przeciążeń 
        f_val = sum(ov.values()) 
        return f_val, loads, ov

    def __lt__(self, other):
        return self.fitness < other.fitness

def crossover(p1, p2):
    # Losowa wymiana genów 
    c1_genes, c2_genes = [], []
    for g1, g2 in zip(p1.genes, p2.genes):
        if random.random() < 0.5:
            c1_genes.append(g1[:]); c2_genes.append(g2[:])
        else:
            c1_genes.append(g2[:]); c2_genes.append(g1[:])
    return ChromosomeDAP(c1_genes), ChromosomeDAP(c2_genes)

def mutate(chrom):
    # Mutacja chromosomu z prawd. p 
    if random.random() > P_CH_MU:
        return chrom
    
    new_genes = [g[:] for g in chrom.genes]
    for i, d in enumerate(DEMANDS):
        # Mutacja genu (przesunięcie jednostki) z prawd. q 
        if random.random() < P_GENE_MU:
            has_flow = [idx for idx, val in enumerate(new_genes[i]) if val > 0]
            if has_flow:
                src = random.choice(has_flow)
                dst = random.randint(0, len(d['paths']) - 1)
                new_genes[i][src] -= 1
                new_genes[i][dst] += 1
    return ChromosomeDAP(new_genes)

def run_ea_dap():
    # P(0) - Inicjalizacja i sortowanie
    pop = sorted([ChromosomeDAP() for _ in range(N)])
    trajectory = [pop[0].fitness]
    
    for n in range(1, MAX_GEN + 1):
        offspring_list = []
        # Losowy dobór par
        for _ in range(K_ops):
            p1, p2 = random.sample(pop, 2)
            c1, c2 = crossover(p1, p2)
            offspring_list.extend([mutate(c1), mutate(c2)])
        
        # P(n) := select_best_N [P(n-1) + O] (N+K EA) [cite: 511, 520, 529]
        pop = sorted(pop + offspring_list)[:N]
        trajectory.append(pop[0].fitness)
        if n % 10 == 0:
            print(f"Generacja {n}: Najlepsza suma przeciążeń = {pop[0].fitness}")

    # WYNIK KOŃCOWY
    best = pop[0]
    print("\n" + "="*20 + " RAPORT KOŃCOWY " + "="*20)
    print(f"Trajektoria: {trajectory}")
    
    print("\nPRZEPŁYWY ŚCIEŻKOWE x(d,p):") 
    for i, d in enumerate(DEMANDS):
        print(f"Demand {d['id']} (vol={d['vol']}): {best.genes[i]}")
    
    print("\nLINK | LOAD | CAP | OVERLOAD") 
    for l in sorted(LINKS_CAP.keys()):
        load = best.link_loads[l]
        cap = LINKS_CAP[l]
        over = best.overloads[l]
        print(f"{l:<4} | {load:<4} | {cap:<3} | {over}")
    
    print(f"\n Wynik funkcji celu: {best.fitness}")

if __name__ == "__main__":
    run_ea_dap()