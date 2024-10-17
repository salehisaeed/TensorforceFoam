import os
import numpy as np
from environment import OpenFOAMEnvironment
from tensorforce import Agent
from postprocess import force_coeffs


def evaluate_model(control):

    environment = OpenFOAMEnvironment()
    environment.params['keep_results'] = True
    model_dir = 'model'

    if(os.path.exists(model_dir + '/checkpoint')):
        print("restore the model")
        agent = Agent.load(directory=model_dir, format='checkpoint', 
                           environment=environment)
    
    print("start simulation, control = " + str(control))
    state = environment.reset()

    n_actions = 600
    for k in range(n_actions):
        if control and k > 100:
            action = agent.act(state, independent=True, deterministic=True)
        else:
            action = np.array([0])
        state, terminal, reward = environment.execute(action)

    environment.close()
    agent.close()


def main():

    evaluate_model(control=True)
    os.system('mv results/history0 results/test/history_controlled')
    os.system('mv cylinder/run0 cylinder/controlled')

    evaluate_model(control=False)
    os.system('mv results/history0 results/test/history_baseline')
    os.system('mv cylinder/run0 cylinder/baseline')

    force_coeffs()


if __name__ == '__main__':
    main()
