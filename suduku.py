import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random, time, threading, copy, json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

SIZE, BLOCK = 9, 3
MAX_FITNESS = 243  # 9 * (rows + cols + blocks)
STAGNANT_LIMIT = 500


def block_index(r, c):
    return (r // BLOCK) * BLOCK + (c // BLOCK)


class SudokuGA:
    def __init__(self, givens, pop_size=500, mut_rate=0.06, crossover_rate=0.9, tournament_k=3, elitism=True):
        self.givens = [row[:] for row in givens]
        self.fixed = [[cell != 0 for cell in row] for row in self.givens]
        self.row_missing = [self._missing_in_row(r) for r in range(SIZE)]

        self.pop_size = int(pop_size)
        self.mut_rate = float(mut_rate)
        self.crossover_rate = float(crossover_rate)
        self.tournament_k = int(tournament_k)
        self.elitism = elitism

        self.population = []
        self.fitnesses = []
        self.best = None
        self.best_fit = -1
        self.generation = 0
        self.history = []

        self.stagnant_count = 0
        self.last_best = self.best_fit
        self.stopped_no_improve = False
        self.valid_solution = False

        self._init_population()

    # ---------------- Population Initialization ----------------
    def _missing_in_row(self, r):
        present = {self.givens[r][c] for c in range(SIZE) if self.givens[r][c] != 0}
        return [d for d in range(1, SIZE + 1) if d not in present]

    def _make_individual(self):
        ind = [row[:] for row in self.givens]
        for r in range(SIZE):
            missing = self.row_missing[r][:]
            random.shuffle(missing)
            idx = 0
            for c in range(SIZE):
                if not self.fixed[r][c]:
                    ind[r][c] = missing[idx]
                    idx += 1
        return ind

    def _init_population(self):
        self.population = [self._make_individual() for _ in range(self.pop_size)]
        self.fitnesses = [self.evaluate(ind) for ind in self.population]
        self._update_best()
        self.generation = 0
        self.history = []
        self.stagnant_count = 0
        self.last_best = self.best_fit

    # ---------------- Fitness & Selection ----------------
    def evaluate(self, ind):
        score = 0
        for r in range(SIZE):
            if len(set(ind[r])) == SIZE:
                score += SIZE
        for c in range(SIZE):
            col = [ind[r][c] for r in range(SIZE)]
            if len(set(col)) == SIZE:
                score += SIZE
        for br in range(0, SIZE, BLOCK):
            for bc in range(0, SIZE, BLOCK):
                block = [ind[r][c] for r in range(br, br + BLOCK) for c in range(bc, bc + BLOCK)]
                if len(set(block)) == SIZE:
                    score += SIZE
        return score

    def _update_best(self):
        for i, f in enumerate(self.fitnesses):
            if f > self.best_fit:
                self.best_fit = f
                self.best = copy.deepcopy(self.population[i])

    def tournament_select(self):
        best = None
        best_fit = -1
        for _ in range(self.tournament_k):
            idx = random.randrange(self.pop_size)
            if self.fitnesses[idx] > best_fit:
                best_fit = self.fitnesses[idx]
                best = self.population[idx]
        return copy.deepcopy(best)

    # ---------------- Genetic Operators ----------------
    def crossover(self, a, b):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(a), copy.deepcopy(b)
        c1, c2 = [row[:] for row in a], [row[:] for row in b]
        for r in range(SIZE):
            if random.random() < 0.5:
                for c in range(SIZE):
                    if not self.fixed[r][c]:
                        c1[r][c], c2[r][c] = c2[r][c], c1[r][c]
        return c1, c2

    def mutate(self, ind):
        for r in range(SIZE):
            if random.random() < self.mut_rate:
                indices = [c for c in range(SIZE) if not self.fixed[r][c]]
                if len(indices) >= 2:
                    c1, c2 = random.sample(indices, 2)
                    ind[r][c1], ind[r][c2] = ind[r][c2], ind[r][c1]

    def repair_columns(self, ind):
        for c in range(SIZE):
            col = [ind[r][c] for r in range(SIZE)]
            missing = [n for n in range(1, SIZE + 1) if n not in col]
            if not missing:
                continue
            duplicates = [(r, c) for r in range(SIZE) if col.count(ind[r][c]) > 1 and not self.fixed[r][c]]
            random.shuffle(missing)
            for rpos, cpos in duplicates:
                for m in missing:
                    if m not in ind[rpos]:
                        ind[rpos][cpos] = m
                        missing.remove(m)
                        break
        return ind

    def local_improve(self, ind, attempts=20):
        best_grid, best_score = ind, self.evaluate(ind)
        for _ in range(attempts):
            a = copy.deepcopy(best_grid)
            r = random.randrange(SIZE)
            idxs = [c for c in range(SIZE) if not self.fixed[r][c]]
            if len(idxs) < 2:
                continue
            c1, c2 = random.sample(idxs, 2)
            a[r][c1], a[r][c2] = a[r][c2], a[r][c1]
            score = self.evaluate(a)
            if score > best_score:
                best_grid, best_score = a, score
        return best_grid

    # ---------------- Evolution Step ----------------
    def step(self):
        new_pop = [copy.deepcopy(self.best)] if self.elitism and self.best else []
        while len(new_pop) < self.pop_size:
            p1, p2 = self.tournament_select(), self.tournament_select()
            c1, c2 = self.crossover(p1, p2)
            self.mutate(c1), self.mutate(c2)
            if random.random() < 0.35: c1 = self.repair_columns(c1)
            if random.random() < 0.35: c2 = self.repair_columns(c2)
            if random.random() < 0.08: c1 = self.local_improve(c1, 10)
            if random.random() < 0.08: c2 = self.local_improve(c2, 10)
            new_pop.extend([c1, c2])
        self.population = new_pop[:self.pop_size]
        self.fitnesses = [self.evaluate(ind) for ind in self.population]
        self._update_best()
        self.history.append(self.best_fit)
        self.generation += 1

    # ---------------- Solution Check ----------------
    def is_valid_solution(self, grid):
        for r in range(SIZE):
            if sorted(grid[r]) != list(range(1, SIZE + 1)): return False
        for c in range(SIZE):
            col = [grid[r][c] for r in range(SIZE)]
            if sorted(col) != list(range(1, SIZE + 1)): return False
        for br in range(0, SIZE, BLOCK):
            for bc in range(0, SIZE, BLOCK):
                block = [grid[r][c] for r in range(br, br + BLOCK) for c in range(bc, bc + BLOCK)]
                if sorted(block) != list(range(1, SIZE + 1)): return False
        return True

    # ---------------- Run GA ----------------
    def run_until(self, max_gen=1000, stop_event=None, pause_event=None, update_callback=None):
        start_time = time.time()
        while self.generation < max_gen and self.best_fit < MAX_FITNESS:
            if stop_event and stop_event.is_set(): break
            if pause_event and pause_event.is_set():
                time.sleep(0.1)
                continue

            self.step()
            if update_callback: update_callback()

            self.stagnant_count = self.stagnant_count + 1 if self.best_fit == self.last_best else 0
            if self.stagnant_count == 0: self.last_best = self.best_fit

            if self.stagnant_count >= STAGNANT_LIMIT:
                self.valid_solution = self.best and self.is_valid_solution(self.best)
                self.stopped_no_improve = True
                break

        return self.best, self.best_fit, self.generation, time.time() - start_time


# -------------------- GUI --------------------
class SudokuGUI:
    def __init__(self, root):
        self.root = root
        root.title('Sudoku GA Solver')
        self.cells = [[None] * SIZE for _ in range(SIZE)]
        self.ga, self.stop_event, self.pause_event = None, None, None
        self.running, self.start_time = False, None
        self._build_ui()

    def _build_ui(self):
        # Sudoku grid
        gridf = ttk.Frame(self.root)
        gridf.grid(row=0, column=0, padx=10, pady=10)
        for r in range(SIZE):
            for c in range(SIZE):
                e = ttk.Entry(gridf, width=3, justify='center', font=('TkDefaultFont', 14))
                e.grid(row=r, column=c, padx=(1 if c % 3 != 0 else 4), pady=(1 if r % 3 != 0 else 4))
                self.cells[r][c] = e

        # Control panel
        ctrl = ttk.Frame(self.root)
        ctrl.grid(row=0, column=1, sticky='n', padx=10, pady=10)
        ttk.Label(ctrl, text='GA Params', font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2)
        self.pop_entry, self.mut_entry, self.cx_entry, self.tourn_entry, self.maxgen_entry = [ttk.Entry(ctrl, width=8) for _ in range(5)]
        defaults = [('Population', self.pop_entry, '500'), ('Mutation', self.mut_entry, '0.06'),
                    ('Crossover', self.cx_entry, '0.9'), ('Tournament k', self.tourn_entry, '3'),
                    ('Max gen', self.maxgen_entry, '10000')]
        for i, (label, entry, val) in enumerate(defaults, 1):
            ttk.Label(ctrl, text=label + ':').grid(row=i, column=0, sticky='w')
            entry.insert(0, val)
            entry.grid(row=i, column=1, sticky='e')

        # Buttons
        ttk.Button(ctrl, text='Start', command=self.start).grid(row=6, column=0, pady=4)
        ttk.Button(ctrl, text='Stop', command=self.stop).grid(row=6, column=1, pady=4)
        ttk.Button(ctrl, text='Pause', command=self.pause).grid(row=7, column=0, pady=4)
        ttk.Button(ctrl, text='Resume', command=self.resume).grid(row=7, column=1, pady=4)
        ttk.Button(ctrl, text='Save Puzzle', command=self.save_puzzle).grid(row=8, column=0, columnspan=2, sticky='ew', pady=4)
        ttk.Button(ctrl, text='Load Puzzle', command=self.load_puzzle).grid(row=9, column=0, columnspan=2, sticky='ew', pady=4)
        ttk.Button(ctrl, text='Upload', command=self.upload_puzzle).grid(row=10, column=0, columnspan=2, sticky='ew', pady=4)

        # Fitness plot
        self.fig, self.ax = plt.subplots(figsize=(3, 2))
        self.ax.set(title='Fitness Evolution', xlabel='Generation', ylabel='Best Fitness')
        self.line, = self.ax.plot([], [], 'b-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)

        # Status labels
        status = ttk.Frame(self.root)
        status.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        self.gen_label = ttk.Label(status, text='Gen: 0')
        self.fit_label = ttk.Label(status, text=f'Best fitness: 0/{MAX_FITNESS}')
        self.time_label = ttk.Label(status, text='Time: 0s')
        self.gen_label.grid(row=0, column=0, sticky='w')
        self.fit_label.grid(row=0, column=1, sticky='w', padx=10)
        self.time_label.grid(row=0, column=2, sticky='w', padx=10)

    # ---------------- Grid Operations ----------------
    def read_grid(self):
        return [[int(self.cells[r][c].get()) if self.cells[r][c].get().isdigit() else 0 for c in range(SIZE)] for r in range(SIZE)]

    def clear_grid(self):
        for r in range(SIZE):
            for c in range(SIZE):
                self.cells[r][c].delete(0, tk.END)

    def save_puzzle(self):
        grid = self.read_grid()
        fn = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON files', '*.json')])
        if fn:
            with open(fn, 'w') as f: json.dump(grid, f)
            messagebox.showinfo('Saved', f'Puzzle saved to {fn}')

    def load_puzzle(self):
        self.clear_grid()
        messagebox.showinfo('Load Puzzle', 'Grid cleared. Please enter your puzzle manually.')

    def is_valid_given_puzzle(self, grid):
        for r in range(SIZE):
            nums = [n for n in grid[r] if n != 0]
            if len(nums) != len(set(nums)): return False
        for c in range(SIZE):
            nums = [grid[r][c] for r in range(SIZE) if grid[r][c] != 0]
            if len(nums) != len(set(nums)): return False
        for br in range(0, SIZE, BLOCK):
            for bc in range(0, SIZE, BLOCK):
                nums = [grid[r][c] for r in range(br, br+BLOCK) for c in range(bc, bc+BLOCK) if grid[r][c] != 0]
                if len(nums) != len(set(nums)): return False
        return True

    # ---------------- GA Control ----------------
    def start(self):
        if self.running: return
        givens = self.read_grid()
        if not self.is_valid_given_puzzle(givens):
            messagebox.showerror("Invalid Sudoku", "Puzzle has duplicates. Correct it before starting.")
            return

        pop, mut, cx, tourn, maxgen = int(self.pop_entry.get()), float(self.mut_entry.get()), float(self.cx_entry.get()), int(self.tourn_entry.get()), int(self.maxgen_entry.get())
        self.ga = SudokuGA(givens, pop_size=pop, mut_rate=mut, crossover_rate=cx, tournament_k=tourn)
        self.stop_event, self.pause_event = threading.Event(), threading.Event()
        self.running, self.start_time = True, time.time()

        def update_callback():
            if not self.pause_event.is_set(): self.root.after(1, self.update_ui)

        def worker():
            best, fit, gen, elapsed = self.ga.run_until(max_gen=maxgen, stop_event=self.stop_event, pause_event=self.pause_event, update_callback=update_callback)
            self.running = False
            self.root.after(1, self.update_ui)
            if self.ga.valid_solution or fit >= MAX_FITNESS:
                self.root.after(1, lambda: messagebox.showinfo('Success', f'Sudoku solved in {gen} generations ({elapsed:.2f}s)'))
            else:
                self.root.after(1, lambda: messagebox.showerror('Failed', f'GA could not solve puzzle after {gen} generations ({elapsed:.2f}s)'))

        threading.Thread(target=worker, daemon=True).start()

    def pause(self): self.pause_event.set() if self.running else None
    def resume(self): self.pause_event.clear() if self.running else None
    def stop(self):
        if self.running:
            self.stop_event.set()
            self.running = False

    def update_ui(self):
        if not self.ga or not self.ga.best: return
        best = self.ga.best
        for r in range(SIZE):
            for c in range(SIZE):
                self.cells[r][c].delete(0, tk.END)
                self.cells[r][c].insert(0, str(best[r][c]))
        self.gen_label.config(text=f'Gen: {self.ga.generation}')
        self.fit_label.config(text=f'Best fitness: {self.ga.best_fit}/{MAX_FITNESS}')
        self.time_label.config(text=f'Time: {int(time.time() - self.start_time)}s')
        self.line.set_data(range(len(self.ga.history)), self.ga.history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def upload_puzzle(self):
        fn = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        if not fn: return
        try:
            with open(fn) as f:
                lines = f.readlines()
            if len(lines) != SIZE: raise ValueError("File must have 9 lines")
            grid = [[int(n) if n.isdigit() else 0 for n in line.strip().split()] for line in lines]
            self.clear_grid()
            for r in range(SIZE):
                for c in range(SIZE):
                    if grid[r][c] != 0: self.cells[r][c].insert(0, str(grid[r][c]))
            messagebox.showinfo('Uploaded', f'Puzzle uploaded from {fn}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to upload puzzle:\n{e}')


if __name__ == '__main__':
    root = tk.Tk()
    gui = SudokuGUI(root)
    root.mainloop()
