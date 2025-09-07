# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:42:33 2024

@author: glshr
"""

import numpy as np
import random

class AntColony:
    def __init__(self, distance_matrix, n_ants, n_iterations, alpha, beta, evaporation_rate, q=1):
        self.distance_matrix = distance_matrix
        self.pheromone = np.ones(distance_matrix.shape) / len(distance_matrix)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.n_nodes = distance_matrix.shape[0]

    def _choose_next_node(self, current_node, visited_nodes):
        probabilities = []
        for i in range(self.n_nodes):
            if i not in visited_nodes:
                prob = (self.pheromone[current_node][i] ** self.alpha) * \
                       ((1 / self.distance_matrix[current_node][i]) ** self.beta)
                probabilities.append(prob)
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return np.random.choice(range(self.n_nodes), p=probabilities)

    def _update_pheromones(self, all_routes, all_costs):
        self.pheromone *= (1 - self.evaporation_rate)
        for route, cost in zip(all_routes, all_costs):
            for i in range(len(route) - 1):
                self.pheromone[route[i]][route[i + 1]] += self.q / cost

    def optimize(self):
        best_cost = float('inf')
        best_route = None

        for iteration in range(self.n_iterations):
            all_routes = []
            all_costs = []

            for ant in range(self.n_ants):
                route = [random.randint(0, self.n_nodes - 1)]
                while len(route) < self.n_nodes:
                    current_node = route[-1]
                    next_node = self._choose_next_node(current_node, route)
                    route.append(next_node)
                route.append(route[0])  # Return to the start
                cost = sum(self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
                all_routes.append(route)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_route = route

            self._update_pheromones(all_routes, all_costs)
            print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

        return best_route, best_cost


# Örnek Kullanım

import numpy as np
import os

def pad_matrix(matrix, fill_value=0):
    max_length = max(len(row) for row in matrix)
    return [row + [fill_value] * (max_length - len(row)) for row in matrix]

def read_matrices_from_files(directory_path, file_prefix="airland", file_extension=".txt"):
    matrices = []
    
    for i in range(1, 14):  # 1'den 13'e kadar dosyaları oku
        file_name = f"{file_prefix}{i}{file_extension}"
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                lines = file.readlines()
                matrix = []
                for line in lines:
                    # Her satırı sayılara dönüştür ve matrise ekle
                    row = list(map(float, line.split()))
                    matrix.append(row)
                
                # Satırları doldurarak eşit uzunluğa getir
                padded_matrix = pad_matrix(matrix)
                matrices.append(np.array(padded_matrix))
        else:
            print(f"Dosya bulunamadı: {file_path}")
    
    return matrices



# Örnek Kullanım

# Matris dosyalarının olduğu dizin
directory_path = r"C:\Users\glshr\OneDrive\Belgeler\ders\ders 4. sınıf\yapay zeka teknikleri\files"  # Örneğin: "/mnt/data/"
file_prefix = "airland"  # Dosya isimleri: airland1.txt, airland2.txt, ...
file_extension = ".txt"

matrices = read_matrices_from_files(directory_path, file_prefix, file_extension)
def validate_matrices(matrices):
    for i, matrix in enumerate(matrices):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matris {i + 1} kare değil! Boyut: {matrix.shape}")

# Tüm matrisleri kontrol et
validate_matrices(matrices)

# İlk matrisi kullanarak AntColony'yi çalıştır
if matrices:
    for i, matrix in enumerate(matrices):
        print(f"Matris {i + 1} için optimizasyon başlıyor...")
        n_ants = 5
        n_iterations = 50
        alpha = 1  # Feromon etkisi
        beta = 2   # Mesafe etkisi
        evaporation_rate = 0.5

        colony = AntColony(matrix, n_ants, n_iterations, alpha, beta, evaporation_rate)
        best_route, best_cost = colony.optimize()

        print(f"Matris {i + 1} - En İyi Yol:", best_route)
        print(f"Matris {i + 1} - En Düşük Maliyet:", best_cost)
else:
    print("Hiçbir dosya okunamadı.")
