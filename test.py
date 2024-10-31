import pandas as pd
import numpy as np
import random

from geopy.distance import geodesic
from itertools import combinations

# 读取城市数据
CITIES = pd.read_csv('cities/italy.csv', header=None, names=['name', 'lat', 'lon'])

# 计算距离矩阵
DIST_MATRIX = np.zeros((len(CITIES), len(CITIES)))
for c1, c2 in combinations(CITIES.itertuples(), 2):
    DIST_MATRIX[c1.Index, c2.Index] = DIST_MATRIX[c2.Index, c1.Index] = geodesic(
        (c1.lat, c1.lon), (c2.lat, c2.lon)
    ).km

# 遗传算法参数
population_size = 100
num_generations = 500
mutation_rate = 0.01

# 初始化种群
def initialize_population(size):
    population = []
    for _ in range(size):
        individual = list(range(len(CITIES)))
        random.shuffle(individual)
        population.append(individual)
    return population

# 计算适应度
def calculate_fitness(individual):
    distance = sum(DIST_MATRIX[individual[i]][individual[i + 1]] for i in range(len(individual) - 1))
    distance += DIST_MATRIX[individual[-1]][individual[0]]  # 回到起点
    return 1 / distance  # 适应度为距离的倒数

# 选择操作
def selection(population):
    fitness_values = [calculate_fitness(individual) for individual in population]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    selected = np.random.choice(population, size=len(population), p=probabilities)
    return selected.tolist()

# 交叉操作
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(CITIES)), 2))
    child = [-1] * len(CITIES)
    child[start:end] = parent1[start:end]
    
    # 填充剩余的城市
    current_pos = end
    for city in parent2:
        if city not in child:
            child[current_pos] = city
            current_pos = (current_pos + 1) % len(CITIES)
            
    return child

# 变异操作
def mutate(individual):
    for i in range(len(CITIES)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(CITIES) - 1)
            individual[i], individual[j] = individual[j], individual[i]

# 主遗传算法过程
def genetic_algorithm():
    population = initialize_population(population_size)
    
    for generation in range(num_generations):
        population = selection(population)
        next_generation = []

        for i in range(0, population_size, 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < population_size else population[0]
            child = crossover(parent1, parent2)
            mutate(child)
            next_generation.append(child)

        population = next_generation

    # 找到最佳个体
    best_individual = min(population, key=lambda ind: 1 / calculate_fitness(ind))
    best_distance = 1 / calculate_fitness(best_individual)
    
    return best_individual, best_distance

# 执行遗传算法
best_path, best_distance = genetic_algorithm()

# 输出结果
best_path_cities = CITIES.iloc[best_path]['name'].tolist()
print("最佳路径:", best_path_cities)
print("最佳距离:", best_distance)

