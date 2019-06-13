import numpy as np
import pymunk as pm
import pymunk.pygame_util
import pygame
import random
from pygame.color import *
from pygame.locals import *
import pybullet as p
import math


NUM_COLORS = 656
COLORS_TUPLE = list(THECOLORS.values())
goal_color = (255, 0, 0, 255)
COLORS_TUPLE.remove((goal_color))

collision_types = {
    "object": 1,
    "player": 2,
    "goal": 3,
    "box": 4,
    "circle": 5,
    "arrow": 6
}


def construct_space(size):
    screen_size = 84
    screen = pygame.Surface((size, size))
    space = pm.Space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    space.gravity = 0, 0

    static_body = space.static_body
    segs = [pm.Segment(static_body, (0.0, 0.0), (0.0, screen_size), 0.0),
            pm.Segment(static_body, (0.0, 0.0), (screen_size, 0.0), 0.0),
            pm.Segment(static_body, (screen_size, 0.0), (screen_size, screen_size), 0.0),
            pm.Segment(static_body, (0.0, screen_size), (screen_size, screen_size), 0.0)]

    for seg in segs:
        seg.elasticity = 0.95
        seg.friction = 0.9

    space.add(segs)

    return space, screen, draw_options


def reset_3d_velocity(bodies, physicsClientId=0):
    for body in bodies:
        vel, _ = p.getBaseVelocity(body, physicsClientId=physicsClientId)
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=physicsClientId)
        v_norm = sum([v**2 for v in list(vel)])

        def oob_x(x):
            return (x < -0.5) or (x > 10.5)

        def oob_y(x):
            return (x < -10.5) or (x > 10.5)

        if v_norm < 0.01 or oob_x(pos[0]) or oob_y(pos[1]):
            x, y = np.random.uniform(0, 10, size=(2))
            vx, vy = np.random.uniform(-5.0, 5.0, size=[2])

            p.resetBasePositionAndOrientation(body, [x, y, 3.0], [0, 0, 0, 1], physicsClientId=physicsClientId)
            p.resetBaseVelocity(body, [vx, vy, 0], physicsClientId=physicsClientId)


def save_bullet_state(physicsClientId=0):
    state = {}

    for i in range(p.getNumBodies()):
        pos, orn = p.getBasePositionAndOrientation(i, physicsClientId=physicsClientId)
        linVel, angVel = p.getBaseVelocity(i, physicsClientId=physicsClientId)

        state[i] = (pos, orn, linVel, angVel)

    return state


def restore_bullet_state(state, physicsClientId=0):

    for i, tup in state.items():
        pos, orn, linVel, angVel = tup
        p.resetBasePositionAndOrientation(i, pos, orn, physicsClientId=physicsClientId)
        p.resetBaseVelocity(i, linVel, angVel, physicsClientId=physicsClientId)


def construct_3d_space(min_obj=7, physicsClientId=0):

    p.setGravity(0, 0, -10, physicsClientId=physicsClientId)
    # First construct floor
    sid = p.createVisualShape(p.GEOM_BOX, halfExtents = [100, 100, 4], rgbaColor=(0.6, 0.2, 0.4, 1), physicsClientId=physicsClientId)
    cid = p.createCollisionShape(p.GEOM_BOX, halfExtents = [100, 100, 4], physicsClientId=physicsClientId)
    wid = p.createMultiBody(baseMass=0, baseVisualShapeIndex = sid, baseCollisionShapeIndex=cid, basePosition = [0, 0, -4], baseOrientation=[0, 0, 0, 1], physicsClientId=physicsClientId)

    p.changeDynamics(wid, -1, restitution=0.99, physicsClientId=physicsClientId)
    # Then construct two walls
    # sid = p.createVisualShape(p.GEOM_BOX, halfExtents = [10000, 10000, 4], rgbaColor=(0.6, 0.6, 0, 1))
    # wid = p.createMultiBody(baseMass=0, baseVisualShapeIndex = sid, basePosition = [14, 0, 0], baseOrientation=[math.sqrt(2), 0, 0, math.sqrt(2)])
    # planeId = p.loadURDF("plane.urdf", basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])

    # Construct objects
    bodies = []
    for i in range(min_obj):
        x, y = np.random.uniform(-10, 10, size=(2))
        vx, vy = np.random.uniform(-5.0, 5.0, size=[2])
        vid = p.createVisualShape(p.GEOM_SPHERE, radius = 0.5, rgbaColor=(0, 0, 1, 1), physicsClientId=physicsClientId)
        object_id = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.5, physicsClientId=physicsClientId)
        wid = p.createMultiBody(baseMass=1, baseVisualShapeIndex = vid, baseCollisionShapeIndex=object_id, basePosition = [abs(x), y, 3], baseOrientation=[0, 0, 0, 1], physicsClientId=physicsClientId)

        bodies.append(wid)
        p.resetBaseVelocity(object_id, [vx, vy, 0], physicsClientId=physicsClientId)
        p.changeDynamics(object_id, -1, restitution=0.99, physicsClientId=physicsClientId)

    return bodies


def construct_3d_shapes_space(min_obj=7, physicsClientId=0):

    p.setGravity(0, 0, -10)
    # First construct floor
    sid = p.createVisualShape(p.GEOM_BOX, halfExtents = [100, 100, 4], rgbaColor=(0.6, 0.2, 0.4, 1), physicsClientId=physicsClientId)
    cid = p.createCollisionShape(p.GEOM_BOX, halfExtents = [100, 100, 4], physicsClientId=physicsClientId)
    wid = p.createMultiBody(baseMass=0, baseVisualShapeIndex = sid, baseCollisionShapeIndex=cid, basePosition = [0, 0, -4], baseOrientation=[0, 0, 0, 1], physicsClientId=physicsClientId)
    p.changeDynamics(wid, -1, restitution=0.99, physicsClientId=physicsClientId)

    # Then construct two walls
    # sid = p.createVisualShape(p.GEOM_BOX, halfExtents = [10000, 10000, 4], rgbaColor=(0.6, 0.6, 0, 1))
    # wid = p.createMultiBody(baseMass=0, baseVisualShapeIndex = sid, basePosition = [14, 0, 0], baseOrientation=[math.sqrt(2), 0, 0, math.sqrt(2)])
    # planeId = p.loadURDF("plane.urdf", basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])

    # Construct objects
    bodies = []
    for i in range(min_obj):
        x, y = np.random.uniform(-10, 10, size=(2))
        vx, vy = np.random.uniform(-5.0, 5.0, size=[2])
        vid = p.createVisualShape(p.GEOM_SPHERE, radius = 0.5, rgbaColor=(0, 0, 1, 1), physicsClientId=physicsClientId)
        object_id = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.5, physicsClientId=physicsClientId)
        wid = p.createMultiBody(baseMass=1, baseVisualShapeIndex = vid, baseCollisionShapeIndex=object_id, basePosition = [abs(x), y, 3], baseOrientation=[0, 0, 0, 1], physicsClientId=physicsClientId)

        bodies.append(wid)
        p.resetBaseVelocity(object_id, [vx, vy, 0], physicsClientId=physicsClientId)
        p.changeDynamics(object_id, -1, restitution=0.99, physicsClientId=physicsClientId)

    goals = []
    for i in range(min_obj):
        x, y = np.random.uniform(-10, 10, size=(2))
        vx, vy = np.random.uniform(-5.0, 5.0, size=[2])
        vid = p.createVisualShape(p.GEOM_SPHERE, halfExtents = [0.5, 0.5, 0.5], rgbaColor=(0.8, 0.4, 1, 1), physicsClientId=physicsClientId)
        object_id = p.createCollisionShape(p.GEOM_SPHERE, halfExtents = [0.5, 0.5, 0.5], physicsClientId=physicsClientId)
        wid = p.createMultiBody(baseMass=1, baseVisualShapeIndex = vid, baseCollisionShapeIndex=object_id, basePosition = [abs(x), y, 3], baseOrientation=[0, 0, 0, 1], physicsClientId=physicsClientId)

        goals.append(wid)
        p.resetBaseVelocity(object_id, [vx, vy, 0], physicsClientId=physicsClientId)
        p.changeDynamics(object_id, -1, restitution=0.99, physicsClientId=physicsClientId)

    return bodies, goals


def get_3d_image(physicsClientId=0):
    # viewMat = p.computeProjectionMatrixFOV(np.pi/3, 1, 0.01, 100)
    viewMat = p.computeViewMatrix((-1, -1, 2), (10, 10, 2), (0, 0, 1))
    projMat = p.computeProjectionMatrixFOV(100, 1, 0.01, 100)

    _, _, im, dp_im, _ = p.getCameraImage(84,84, viewMatrix=viewMat, physicsClientId=physicsClientId)

    # im[:, :, 1] = (im[:, :, 1] + im[:, :, 2]) / 2

    # Preprocess depth image
    # dp_im = dp_im * 255.
    # im = np.concatenate([im[:, :, :2], dp_im.reshape(84, 84, 1)], axis=2)
    im = im[:, :, :3]
    return im

def spawn_rand_goal_3d(first=False, physicsClientId=0):
    goal_vid =  p.createVisualShape(p.GEOM_BOX, halfExtents = [0.5, 0.5, 0.5], rgbaColor=(0.5, 0, 0.5, 1), physicsClientId=physicsClientId)

    if first:
        x, y = 7, 5
    else:
        x, y = np.random.randint(0, 11, size=[2])
    goal_body = p.createMultiBody(0, baseVisualShapeIndex = goal_vid, basePosition = [x, y, 0.5], baseOrientation=[0, 0, 0, 1], physicsClientId=physicsClientId)

    return goal_body

def respawn_rand_goal(body, physicsClientId=0):
    x, y = np.random.randint(0, 11, size=[2])
    pos, ori = p.getBasePositionAndOrientation(body, physicsClientId=physicsClientId)
    p.resetBasePositionAndOrientation(body, [x, y, 0.5], ori, physicsClientId=physicsClientId)



def step_3d_simulation(player_body, nsteps=1, physicsClientId=0):

    done = False
    for i in range(nsteps):
         p.stepSimulation(physicsClientId=physicsClientId)


    vel, _ = p.getBaseVelocity(player_body, physicsClientId=physicsClientId)
    vel = vel[:2]
    tot = sum([abs(v) for v in list(vel)])

    if tot > 1:
        done = True
    else:
        done = False

    return done


def step_3d_simulation_shooter(player_body, bullet_body, goals, bodies, nsteps=1, physicsClientId=0):

    done = False
    hit = False
    reward = 0
    for i in range(nsteps):
         p.stepSimulation(physicsClientId=physicsClientId)
         bullet_aabb = p.getAABB(bullet_body, physicsClientId=physicsClientId)
         obs = p.getOverlappingObjects(*bullet_aabb, physicsClientId=physicsClientId)

         if obs is not None:
             obs = [ob[0] for ob in obs]
             for ob in obs:
                 if ob in goals:
                    x, y = np.random.uniform(0, 10, size=(2))
                    vx, vy = np.random.uniform(-20.0, 20.0, size=[2])

                    p.resetBasePositionAndOrientation(ob, [x, y, 3.0], [0, 0, 0, 1], physicsClientId=physicsClientId)
                    p.resetBaseVelocity(ob, [vx, vy, 0], physicsClientId=physicsClientId)

                    reward += 1
                    hit = True

                 elif ob in bodies:
                    x, y = np.random.uniform(0, 10, size=(2))
                    vx, vy = np.random.uniform(-20.0, 20.0, size=[2])

                    p.resetBasePositionAndOrientation(ob, [x, y, 3.0], [0, 0, 0, 1], physicsClientId=physicsClientId)
                    p.resetBaseVelocity(ob, [vx, vy, 0], physicsClientId=physicsClientId)

                    reward -= 1
                    hit = True

    if hit:
        p.resetBasePositionAndOrientation(bullet_body, [30, 30, 0.5], [0, 0, 0, 1], physicsClientId=physicsClientId)

    vel, _ = p.getBaseVelocity(player_body, physicsClientId=physicsClientId)
    vel = vel[:2]
    tot = sum([abs(v) for v in list(vel)])

    if tot > 1:
        done = True
    else:
        done = False

    return done, reward



def spawn_rand_goal(space, dim):
    pos_low = 0.1
    pos_high = 0.9
    # pos = dim * random.uniform(pos_low, pos_high), dim * random.uniform(pos_low, pos_high)
    # Initialize at set point for hopefully easier learning
    pos = 40, 6

    body = pm.Body(1, 1, pm.Body.KINEMATIC)
    body.position = pos
    body.velocity = (0, 0)

    shape = pm.Circle(body, 8)
    shape.collision_type = collision_types['goal']

    shape.color = goal_color
    space.add(body, shape)

    return body


def construct_rand_body(space, dim):
    v_scale = 300
    pos_low = 0.2
    pos_high = 0.95

    pos = dim * random.uniform(pos_low, pos_high), dim * random.uniform(pos_low, pos_high)
    velocity = v_scale * random.uniform(-1, 1), v_scale * random.uniform(-1, 1)
    body = pm.Body(1, 1)
    body.velocity = velocity
    body.position = pos

    # size_a, size_b = np.random.randint(3, 7, size=(2,))
    size_a, size_b = 5, 5

    if random.randint(0, 2):
        shape = pm.Circle(body, size_a)
    else:
        shape = pm.Poly.create_box(body, size=(size_a, size_b))

    shape.elasticity = 0.95
    shape.friction = 0.9
    shape.collision_type = collision_types['object']

    idx = np.random.randint(1, NUM_COLORS)
    # shape.color = tuple(COLORS_TUPLE[idx])
    # space.color = (0, 255, 0, 0)

    space.add(body, shape)


def construct_rand_box(space, dim):
    # Constructs a box type object
    v_scale = 300
    pos_low = 0.2
    pos_high = 0.95

    pos = dim * random.uniform(pos_low, pos_high), dim * random.uniform(pos_low, pos_high)
    velocity = v_scale * random.uniform(-1, 1), v_scale * random.uniform(-1, 1)
    body = pm.Body(1, 1)
    body.velocity = velocity
    body.position = pos

    # size_a, size_b = np.random.randint(3, 7, size=(2,))
    size_a, size_b = 5, 5

    shape = pm.Poly.create_box(body, size=(size_a, size_b))

    shape.elasticity = 0.95
    shape.friction = 0.9
    shape.collision_type = collision_types['box']

    idx = np.random.randint(1, NUM_COLORS)
    # shape.color = tuple(COLORS_TUPLE[idx])
    # space.color = (0, 255, 0, 0)

    space.add(body, shape)


def construct_rand_circle(space, dim):
    # Constructs a box type object
    v_scale = 300
    pos_low = 0.2
    pos_high = 0.95

    pos = dim * random.uniform(pos_low, pos_high), dim * random.uniform(pos_low, pos_high)
    velocity = v_scale * random.uniform(-1, 1), v_scale * random.uniform(-1, 1)
    body = pm.Body(1, 1)
    body.velocity = velocity
    body.position = pos

    # size_a, size_b = np.random.randint(3, 7, size=(2,))
    size_a, size_b = 5, 5

    shape = pm.Circle(body, size_a)

    shape.elasticity = 0.95
    shape.friction = 0.9
    shape.collision_type = collision_types['circle']

    idx = np.random.randint(1, NUM_COLORS)
    # shape.color = tuple(COLORS_TUPLE[idx])
    space.color = (0, 255, 0, 0)

    space.add(body, shape)


def construct_rand_wall(space, dim):
    start_seg = dim * np.random.uniform(0.1, 0.9, size=(2,))
    if random.randint(0, 1):
        # construct straight wall
        end = dim * np.random.uniform(0.1, 0.5)
        if random.randint(0, 1):
            end_seg = (start_seg[0], start_seg[1] + end)
        else:
            end_seg = (start_seg[0] + end, start_seg[1])
    else:
        # construct diagonal wall
        end1 = dim * np.random.uniform(0.1, 0.5)
        end2 = dim * np.random.uniform(0.1, 0.5)
        end_seg = (start_seg[0] + end1, start_seg[1] + end2)

    static_body = space.static_body
    seg = pm.Segment(static_body, start_seg, end_seg, 0.0)
    seg.elasticity = 0.95
    seg.friction = 0.9

    space.add(seg)


def construct_default_space(dim=84, min_obj=4, no_object=False):
    space, screen, draw_options = construct_space(dim)
    obj_num = np.random.randint(min_obj, min_obj+3)
    wall_num = np.random.randint(1, 4)
    # scene_color = tuple(np.random.randint(0, 255, size=(3,)))
    scene_color = (0, 0, 0)


    if not no_object:
        for i in range(obj_num):
           construct_rand_body(space, dim)

    for i in range(wall_num):
        construct_rand_wall(space, dim)

    return screen, space, draw_options


def step_simulation(screen, space, draw_options, scene_color=(0, 0, 0)):
    screen.fill(scene_color)
    space.debug_draw(draw_options)
    im = pygame.surfarray.array3d(screen)
    return im


def save_state(space):
    bodies = space._get_bodies()
    snapshot = {}

    for body in bodies:
        snapshot[body] = (body.velocity, body.position, body.angle, body.angular_velocity)

    return snapshot


def restore_state(snapshot):
    for body, val in snapshot.items():
        body.velocity, body.position, body.angle, body.angular_velocity = val
        # Make sure nothing dies during simulation
        body.done = False


def distance(b1, b2):
    dist = ((b1.position[0] - b2.position[0]) ** 2 + (b1.position[1] - b2.position[1]) ** 2) ** 0.5
    return dist
