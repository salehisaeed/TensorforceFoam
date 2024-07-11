from mpi4py import MPI
from tensorforce import Agent
from environment import OpenFOAMEnvironment
from pathlib import Path
import numpy as np
import time, json, os
from train_agent import *

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    time.sleep(3*rank) # To prevent the processes intervention in creating model and run folders
    
    with open('configs/environment.json') as env_json,\
         open('configs/agent.json') as agent_json,\
         open('configs/trainer.json') as trainer_json:
        env_dict = json.load(env_json)
        agent_dict = json.load(agent_json)
        trainer_dict = json.load(trainer_json)

    # Create the environment
    environment = OpenFOAMEnvironment(env_dict, rank)

    # Create or load the DRL agent only for the master process
    if rank == 0:
        load_agent = False # Load previous saved model if available
        checkpoint = Path(agent_dict['saver']['directory'] + '/checkpoint')
        if load_agent and checkpoint.is_file():
            print("Restore the latest saved model")
            agent = Agent.load(
                directory=agent_dict['saver']['directory'], format='checkpoint',
                environment=environment
            )
        else:
            agent = Agent.create(
                agent=agent_dict, environment=environment,
                max_episode_timesteps=env_dict['n_actions']
            )
        # Save current agent mdoel to be loaded and used in OpenFOAM simulations
        model_save_dir = environment.base_policy_dir
        environment.clean_base_model()
        agent.save(directory=model_save_dir, format='saved-model')
        num_updates = 0

        ep_all_str = 'episodes-all'
        ep_train_str = 'episodes-train'
        for dir in [ep_all_str, ep_train_str]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    comm.Barrier()

    n_epochs = trainer_dict['n_epochs']
    for epoch in range(n_epochs):
        if rank == 0:
            print('Epoch {}, started'.format(epoch))
        episode = epoch*nprocs + rank

        tic = time.time()
        environment.reset()
        environment.run()
        states, actions, rewards, terminals = environment.postprocess()
        environment.clean_up()
        toc = time.time()
        sim_time = (toc - tic)/60
        if actions.size != 0:        
            print(('Episode {} completed, run time={:.1f} min, return={:.4f}').format(
                episode, sim_time, np.sum(rewards)))
        else:
            print(('Episode {} did not complete successfully').format(episode))            
        
        states_list = comm.gather(states, root=0)
        actions_list = comm.gather(actions, root=0)
        rewards_list = comm.gather(rewards, root=0)
        terminals_list = comm.gather(terminals, root=0)
        ep_list = comm.gather(episode, root=0)

        if rank == 0:
            for p in range(nprocs):
                # Feed the recorded experience of each episode to the agent and update it
                # Use the data only if CFD has converged
                if actions.size != 0:
                    np.savez_compressed(
                        file=os.path.join(ep_all_str, 'episode-{:04d}.npz'.format(ep_list[p])),
                        states=states_list[p], actions=actions_list[p],
                        terminal=terminals_list[p],reward=rewards_list[p]
                    )

            prepare_training_data(ep_all_str, ep_train_str, agent_dict['batch_size'])
            experience_update(agent, ep_train_str)
            num_updates += 1
            environment.clean_base_model()
            agent.save(directory=model_save_dir, format='saved-model')
            print(('Number of agent updates={}\n').format(num_updates))

        # Sychronize the training loop.
        comm.Barrier()

if __name__ == '__main__':
    main()