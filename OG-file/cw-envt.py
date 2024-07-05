import pybullet as p
import pybullet_data
import time
import numpy as np
import random
from creature import Creature
from population import Population
import math

def make_mountain(num_rocks=100, max_size=0.25, arena_size=10, mountain_height=5):
    def gaussian(x, y, sigma=arena_size/4):
        """Return the height of the mountain at position (x, y) using a Gaussian function."""
        return mountain_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

    for _ in range(num_rocks):
        x = random.uniform(-1 * arena_size/2, arena_size/2)
        y = random.uniform(-1 * arena_size/2, arena_size/2)
        z = gaussian(x, y)  # Height determined by the Gaussian function

        # Adjust the size of the rocks based on height. Higher rocks (closer to the peak) will be smaller.
        size_factor = 1 - (z / mountain_height)
        size = random.uniform(0.1, max_size) * size_factor

        orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
        rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
        rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation)

def make_arena(arena_size=10, wall_height=1):
    wall_thickness = 0.1
    # Floor
    p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, 0)
    
    # Walls
    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

    # Create four walls
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2])

def fitness_function(creature_id, environment_id):
    start_pos, _ = p.getBasePositionAndOrientation(creature_id)
    for _ in range(240 * 30):  # Simulate for 30 seconds
        p.stepSimulation()
        time.sleep(1/240)
    end_pos, _ = p.getBasePositionAndOrientation(creature_id)
    distance_moved = np.linalg.norm(np.asarray(end_pos) - np.asarray(start_pos))
    return end_pos[2]  # Fitness based on height climbed

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    arena_size = 20
    make_arena(arena_size=arena_size)
    make_mountain(arena_size=arena_size)

    p.setGravity(0, 0, -10)

    # Load sandbox + mountain environment here
    # Assuming sandbox + mountain environment is a URDF or SDF file
    mountain_position = (0, 0, -1)  # Adjust as needed
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    p.setAdditionalSearchPath('shapes/')
    mountain = p.loadURDF("gaussian_pyramid.urdf", mountain_position, mountain_orientation, useFixedBase=1)

    # Initialize Population
    pop_size = 20
    population = Population(pop_size, gene_count=10)

    generations = 50
    for generation in range(generations):
        fitness_scores = []
        for creature in population.creatures:
            # Save creature to URDF
            with open('creature.urdf', 'w') as f:
                f.write(creature.to_xml())

            # Load creature into simulation
            creature_id = p.loadURDF('creature.urdf')
            p.resetBasePositionAndOrientation(creature_id, [0, 0, 2.5], [0, 0, 0, 1])

            # Evaluate fitness
            fitness = fitness_function(creature_id, None)
            fitness_scores.append(fitness)

            # Remove creature from simulation
            p.removeBody(creature_id)

        # Update population with fitness scores
        population.update(fitness_scores)

    # After the simulation, you can save the population or the best creature
    best_creature = population.get_best_creature()
    with open('best_creature.urdf', 'w') as f:
        f.write(best_creature.to_xml())

    p.disconnect()

if __name__ == "__main__":
    main()
