import numpy as np
import tensorflow as tf
from matrpo.trainer.agent_model import AgentModel
from gym import spaces

def Policy(env, world, network, nbatch, mode, rho, max_kl, ent_coef, vf_stepsize, vf_iters,
           cg_damping, cg_iters, lbfgs_iters, load_path, **network_kwargs):
    policies = []
    for index, agent in enumerate(world.policy_agents):
        agent.id = index
        agent.comm_matrix = world.comm_matrix[world.comm_matrix[:,index]!=0]
        agent.comms = agent.comm_matrix[:,index][:,None][:,None]
        agent.neighbors = [i for i in np.where(agent.comm_matrix!=0)[1] if i != index]
        agent.observation_space = env.observation_space[index]
        mode_hash = {
            'matrpo' :   cooperative_action_space,
            'central':   cooperative_action_space,
            'trpo'   :   independent_action_space
        }
        agent = mode_hash[mode](agent, env, world)
        model = AgentModel(agent, network, nbatch, rho, max_kl, ent_coef, vf_stepsize, vf_iters,
                           cg_damping, cg_iters, lbfgs_iters, load_path, **network_kwargs)
        policies.append(model)

    return policies

def cooperative_action_space(agent, env, world):
    all_action_spaces = []
    s_index, t_index = 0, 0
    for i, other in enumerate(world.policy_agents):
        space_list, num_acs = split_ac_space(env.action_space[i])
        t_index += num_acs
        all_action_spaces.extend(space_list)
        # store agent's action index
        if other is agent:
            agent.action_index = list(range(s_index, t_index))
        s_index = t_index

    # Augment agent's action space to include parterners' action spaces
    if len(all_action_spaces) > 1:
        agent.action_space = spaces.Tuple(all_action_spaces)
        agent.action_size = t_index
        agent.nmates = len(world.policy_agents)
    else:
        agent.action_space = all_action_spaces[0]
        agent.nmates = agent.action_size = 1

    return agent

def independent_action_space(agent, env, world):
    s_index, t_index = 0, 0
    for i, other in enumerate(world.policy_agents):
        _, num_acs = split_ac_space(env.action_space[i])
        t_index += num_acs
        # store agent's action index
        if other is agent:
            agent.action_index = list(range(s_index, t_index))
            agent.action_space = env.action_space[i]
            agent.nmates = agent.action_size = 1
        s_index = t_index

    return agent

def split_ac_space(ac_space):
    t_index = 0
    space_list = []
    if isinstance(ac_space, spaces.Tuple):
        space_list.extend(ac_space.spaces)
        for space in ac_space.spaces:
            if isinstance(space, spaces.Discrete):
                t_index += 1
            else:
                t_index += space.shape[0]
    elif isinstance(ac_space, spaces.Discrete):
        space_list.append(ac_space)
        t_index += 1
    else:
        space_list.append(ac_space)
        t_index += ac_space.shape[0]

    return space_list, t_index