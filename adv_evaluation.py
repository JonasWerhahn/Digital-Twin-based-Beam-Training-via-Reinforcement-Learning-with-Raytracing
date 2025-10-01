import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate

def read_log(name):
    log_actions = [[]]
    log_states = [[]]
    log_rewards = [[]]

    log = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Results/{name}_log.txt"), "r")
    lines = log.readlines()
    log.close()

    line = lines[0]
    training_episodes = line.split()[0]
    lines.pop(0)
    lines.pop(0)

    current_episode = 0

    for line in lines:
        if line.rstrip():
            values = line.split()
            log_actions[current_episode].append([float(values[0]), float(values[1])])
            log_states[current_episode].append([float(values[2]), float(values[3]), int(values[4]), int(values[5]), float(values[6]), float(values[7])])
            log_rewards[current_episode].append(float(values[8]))
        else:
            current_episode += 1
            log_actions.append([])
            log_states.append([])
            log_rewards.append([])
            
    return log_actions, log_states, log_rewards
        
def avg_reward_of_episode(method, episode):
    avg = 0.0
    num_steps = len(rewards[method][episode])
    for step in range(num_steps):
        avg += rewards[method][episode][step]
    avg /= num_steps
    return avg

def avg_brightness_of_episode(method, episode):
    avg = 0.0
    num_steps = len(states[method][episode])
    for step in range(num_steps):
        avg += states[method][episode][step][4]
    avg /= num_steps
    return avg

def observation_heat_map(method):
    x_scale = 20
    z_scale = 90
            
    data = np.zeros(dtype=int, shape=(x_scale, z_scale))
    for ep in range(len(states[method])):
        for step in range(len(states[method][ep])):
            x_rot = states[method][ep][step][0]*x_scale
            z_rot = states[method][ep][step][1]*z_scale
            data[int(x_rot)][int(z_rot)] += 1
        
    plt.title(f"{agent_names[method]}-Agent: Beobachtungen", fontsize = 18)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.xlabel(f'z-Rotation in {360/z_scale}°', fontsize = 12)
    plt.ylabel(f'x-Rotation in {80/x_scale}°', fontsize = 12)
    plt.colorbar()
    plt.show()
    
def action_heat_map(method):
    action_bound = 24
    scale = action_bound*2+1
            
    data = np.zeros(dtype=int, shape=(scale, scale))
    for ep in range(len(actions[method])):
        for step in range(len(actions[method][ep])):
            x_rot = actions[method][ep][step][0]
            z_rot = actions[method][ep][step][1]
            data[int(x_rot)+action_bound][int(z_rot)+action_bound] += 1

    x_labels = np.arange(-action_bound, action_bound, 4)
    y_labels = np.arange(-action_bound, action_bound, 4)
    plt.xticks(np.linspace(0, data.shape[1]-1, len(x_labels)), x_labels)
    plt.yticks(np.linspace(0, data.shape[0]-1, len(y_labels)), y_labels)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Farblegende hinzufügen
    plt.xlabel('z-Rotation', fontsize = 20)
    plt.ylabel('x-Rotation', fontsize = 20)
    plt.title(f"{agent_names[method]} Actions", fontsize = 24)
    plt.show()
    
def show_graphs():
    avg_rewards = []
    size = 500
    training_episodes = min(len(states[0]), max_episodes)
    print(f"{training_episodes} episodes")
    training_episodes -= training_episodes%size
    for method in range(len(rewards)):
        avg_rewards.append([])
        for ep in range(training_episodes):
            if ep%size==0:
                avg_rewards[method].append(0.0)
            avg_rewards[method][ep//size] += avg_reward_of_episode(method, ep)/size
        
    x = []
    for i in range(training_episodes//size):
        x.append((i+1)*size)
        

    spls = []
    for method in range(len(avg_rewards)):
        spls.append(interpolate.interp1d(x, avg_rewards[method], kind='cubic'))

    x_fine = np.linspace(min(x), max(x), training_episodes//size)
    for method in range(len(avg_rewards)):
        y_smooth = spls[method](x_fine)
        plt.plot(x_fine, y_smooth, '-', label=f"{agent_names[method]}-Agent", linewidth = 2)
            
    plt.tight_layout()
    plt.xlabel('Trainingsepisoden', fontsize = 20)
    plt.ylabel(f"Durchschnittliche Belohnungen über {size} Episoden", fontsize = 20)
    #plt.title(f"Trainingsergebnisse", fontsize = 24)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 20)
    plt.show()

def show_consecutive_rewards():
    avg_cons_rewards = []
    size = 500
    training_episodes = min(len(rewards[0]), max_episodes)
    training_episodes -= training_episodes%size
    
    for method in range(len(rewards)):
        avg_steps = 0
        avg_cons_rewards.append([])
        for ep in range(training_episodes):
            avg_steps += len(rewards[method][ep])
            if ep%size == 0:
                if ep > 0: avg_cons_rewards[method][(ep//size)-1] /= counter
                avg_cons_rewards[method].append(0.0)
                counter = 0
            step = 0
            while rewards[method][ep][step]==1.0:
                step += 1
                if step >= len(rewards[method][ep]): break

            avg_cons_rewards[method][ep//size] += step
            if step>0: 
                counter += 1
        avg_cons_rewards[method][-1] /= counter
        print(f"avg steps {avg_steps/40000}")
    
    x = []
    for i in range(training_episodes//size):
        x.append((i+1)*size)

    spls = []
    for method in range(len(avg_cons_rewards)):
        spls.append(interpolate.interp1d(x, avg_cons_rewards[method], kind='cubic'))

    x_fine = np.linspace(min(x), max(x), training_episodes//size)
    for method in range(len(avg_cons_rewards)):
        y_smooth = spls[method](x_fine)
        plt.plot(x_fine, y_smooth, '-', label=f"{agent_names[method]}-Agent", linewidth = 2)
            
    plt.tight_layout()
    plt.xlabel('Trainingsepisoden', fontsize = 20)
    plt.ylabel(f"Aufeinanderfolgende Belohnungen", fontsize = 20)
    #plt.title(f"Trainingsergebnisse", fontsize = 24)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 20)
    plt.show()

def show_max_brightnesses():
    avg_brightnesses = []
    size = 500
    training_episodes = min(len(states[0]), max_episodes)
    training_episodes -= training_episodes%size
    for method in range(len(states)):
        avg_brightnesses.append([])
        for ep in range(training_episodes):
            if ep%size==0:
                avg_brightnesses[method].append(0.0)
            avg_brightnesses[method][ep//size] += avg_brightness_of_episode(method, ep)/size
        
    x = []
    for i in range(training_episodes//size):
        x.append((i+1)*size)
        

    spls = []
    for method in range(len(avg_brightnesses)):
        spls.append(interpolate.interp1d(x, avg_brightnesses[method], kind='cubic'))

    x_fine = np.linspace(min(x), max(x), training_episodes//size)
    for method in range(len(avg_brightnesses)):
        y_smooth = spls[method](x_fine)
        plt.plot(x_fine, y_smooth, '-', label=f"{agent_names[method]}-Agent", linewidth = 2)
            
    plt.tight_layout()
    plt.xlabel('Trainingsepisoden', fontsize = 20)
    plt.ylabel(f"Maximale Helligkeit der Bilder", fontsize = 20)
    #plt.title(f"Trainingsergebnisse", fontsize = 24)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 20)
    plt.show()
    

log_names = ["adv_ddpg_la6lc4", "v3_ddpg"]
#["adv_ddpg_la4lc4", "adv_ddpg_la4lc6", "adv_ddpg_la6lc4", "adv_ddpg_la6lc6", "adv_ddpg_hl2nf1", "adv_ddpg_hl2nf2", "adv_ddpg_hl3nf2", "adv_ddpg_hl4nf05", "adv_ddpg_hl4nf1", "adv_ddpg_hl4nf2", "adv_ddpg"]
agent_names = ["DDPG LA6LC4", "Env 3 DDPG"]
#["la4lc4", "la4lc6", "la6lc4", "la6lc6", "hl2nf1", "hl2nf2", "hl3nf2", "hl4nf05", "hl4nf1", "hl4nf2", "original"]
actions = []
states = []
rewards = []
for name in log_names:
    log_actions, log_rotations, log_rewards = read_log(name)

    actions.append(log_actions)
    states.append(log_rotations)
    rewards.append(log_rewards)
    
    print(f"read {name}")

for method in range(len(agent_names)):
    print(agent_names[method])
    #observation_heat_map(method)
    #action_heat_map(method)
max_episodes = 40000
show_graphs()
#show_max_brightnesses()
show_consecutive_rewards()