import random
import copy
import statistics
import matplotlib.pyplot as plt

# --- PARAMETRY ---
N = 20          
K_ops = 10      
P_CH_MU = 0.1   
P_GENE_MU = 0.1 
MAX_GEN = 10  
TEST_REPETITIONS = 10 

# --- DANE SIECI ---
MODULE_CAPACITY = 1
LINKS = {1: {'cap_dap': 4, 'cost_ddap': 1}, 2: {'cap_dap': 4, 'cost_ddap': 1},
         3: {'cap_dap': 2, 'cost_ddap': 1}, 4: {'cap_dap': 4, 'cost_ddap': 1},
         5: {'cap_dap': 4, 'cost_ddap': 1}}
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
                f = [0] * len(d['paths'])
                for _ in range(d['vol']): f[random.randint(0, len(f)-1)] += 1
                self.genes.append(f)
        else: self.genes = genes
        self.fitness = self.evaluate()

    def evaluate(self):
        loads = {l: 0 for l in LINKS}
        for i, d in enumerate(DEMANDS):
            for p_idx, flow in enumerate(self.genes[i]):
                for l_id in d['paths'][p_idx]: loads[l_id] += flow
        if self.mode == 'DAP':
            return sum(max(0, loads[l] - LINKS[l]['cap_dap']) for l in LINKS)
        return sum(((loads[l] + MODULE_CAPACITY - 1) // MODULE_CAPACITY) * LINKS[l]['cost_ddap'] for l in LINKS)

    def __lt__(self, other): return self.fitness < other.fitness

def run_ea_full(mode, sel_meth, mut_meth):
    pop = sorted([Chromosome(mode) for _ in range(N)])
    history = [pop[0].fitness]
    
    for _ in range(MAX_GEN):
        O = []
        for _ in range(K_ops):
            # Selekcja
            if sel_meth == 'tournament': p1, p2 = min(random.sample(pop, 2)), min(random.sample(pop, 2))
            elif sel_meth == 'ranking': 
                w = list(range(len(pop), 0, -1))
                p1, p2 = random.choices(pop, weights=w, k=2)
            else: p1, p2 = random.sample(pop, 2)
            
            # Krzyżowanie (Uniform)
            c1_g, c2_g = [], []
            for g1, g2 in zip(p1.genes, p2.genes):
                if random.random() < 0.5:
                    c1_g.append(copy.deepcopy(g1)); c2_g.append(copy.deepcopy(g2))
                else:
                    c1_g.append(copy.deepcopy(g2)); c2_g.append(copy.deepcopy(g1))
            
            # Mutacja
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
        history.append(pop[0].fitness)
    return history, pop[0].fitness

# --- BENCHMARK I REJESTRACJA DANYCH ---
methods_sel = ['random', 'tournament', 'ranking']
methods_mut = ['shift', 'swap']
modes = ['DAP', 'DDAP']

# ... (reszta kodu bez zmian) ...

all_histories = {mode: {} for mode in modes}
table_data = {mode: [] for mode in modes}

for mode in modes:
    for sel in methods_sel:
        for mut in methods_mut:
            label = f"{sel} | {mut}"
            run_results_fitness = []
            run_results_histories = []
            
            for _ in range(TEST_REPETITIONS):
                hist, final_f = run_ea_full(mode, sel, mut)
                run_results_fitness.append(final_f)
                run_results_histories.append(hist)
            
            # 1. Obliczamy dane do tabeli
            avg_f = statistics.mean(run_results_fitness)
            best_f = min(run_results_fitness)
            table_data[mode].append((sel, mut, avg_f, best_f))
            
            # 2. ZNAJDOWANIE NAJLEPSZEJ TRAJEKTORII (zamiast średniej)
            # Szukamy indeksu biegu, który osiągnął best_f
            best_run_idx = run_results_fitness.index(best_f)
            best_history = run_results_histories[best_run_idx]
            
            all_histories[mode][label] = best_history

# --- WYKRESY (teraz pokażą dojście do 2) ---
plt.figure(figsize=(14, 7))
for i, mode in enumerate(modes, 1):
    plt.subplot(1, 2, i)
    for label, history in all_histories[mode].items():
        # Rysujemy linię dla najlepszego przebiegu
        plt.plot(history, label=label, linewidth=2)
    
    plt.title(f"Najlepsza trajektoria optymalizacji: {mode}")
    plt.xlabel("Generacja")
    plt.ylabel("Fitness (Best of 10 runs)")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Ustawienie osi Y, żeby lepiej widzieć dół (opcjonalnie)
    if mode == 'DAP':
        plt.ylim(bottom=0) 

plt.tight_layout()
plt.show()