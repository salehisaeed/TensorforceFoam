from tensorforce import Environment
import numpy as np
import os, time
import subprocess
from mpi4py import MPI

#TODO: add the ramping time and other params of BC to the env json and automate it

class OpenFOAMEnvironment(Environment):
    """
    OpenFOAM Environment for Tensorforce
    """
    def __init__(self, params, rank):
        self.params = params
        self.epoch_number = 0
        self.case_number = rank
        self.base_case = self.params['name'] + '/' + self.params['base_case']
        self.base_policy_dir = self.base_case + '/' + self.params['policy_dir']
        self.start_time = 0
        self.end_time = self.params['action_time_step']*self.params['n_actions']
        self.force_average_steps = int(self.params['force_average_time']/self.params['CFD_time_step'])

    def states(self):
        return dict(
            type='float',
            shape=(self.params['n_states'],)
        )

    def actions(self):
        return dict(
            type='float',
            min_value=-self.params['action_bound'],
            max_value=self.params['action_bound']
        )

    def reset(self):
        self.run_case = self.params['name'] + '/' + self.params['run_cases'] \
                + str(self.case_number) + '_ep' + str(self.epoch_number)
        self.converged = False
        self.lift = np.empty((0), float)
        self.drag = np.empty((0), float)
        self.clean_up()
        subprocess.Popen('cp -r ' + self.base_case + ' ' + self.run_case, shell=True).wait()
        with open(self.run_case + '/log.Allrun.pre', 'a+') as logf:
            subprocess.Popen(
                './' + self.run_case + '/Allrun.pre', shell=True, stdout=logf, stderr=logf
            ).wait()

    def clean_up(self):
        if os.path.isdir(self.run_case):
            subprocess.Popen('rm -rf ' + self.run_case, shell=True).wait()

    def clean_base_model(self):
        subprocess.Popen('rm -rf ' + self.base_policy_dir, shell=True).wait()

    def run(self):
        self.configure_case()
        self.run_OpenFOAM()

    def postprocess(self):
        self.read_results()
        self.compute_reward()
        self.epoch_number += 1
        return self.state_values, self.action_values, self.reward_values, self.terminals

    def configure_case(self):
        self.set_foam_dict('system/controlDict', 'endTime', self.end_time)
        self.set_foam_dict('system/controlDict', 'writeInterval', self.end_time)
        self.set_foam_dict('system/controlDict', 'deltaT', self.params['CFD_time_step'])    

    def run_OpenFOAM(self):
        if self.params['parallel']:
            # Get the number of subdomainis
            output = subprocess.check_output([
                "foamDictionary", "-entry", "numberOfSubdomains","-value",
                self.run_case + "/system/decomposeParDict"
            ])
            self.n_subdomains = int(output.split(b'\n')[0])            
            solver_args = ['-c','pimpleFoam -parallel -case ' + self.run_case + ' > ' 
                        + self.run_case + '/log.pimpleFoam 2>&1']
            comm_spawn = MPI.COMM_SELF.Spawn('sh', args=solver_args, maxprocs=self.n_subdomains)
            # comm_spawn = MPI.COMM_SELF.Spawn_multiple('sh', args=solver_args, maxprocs=n_sub_domains)
            
            # wait until the final time-step folder is created          
            self.waiter()
            comm_spawn.Free()
            # comm_spawn.Disconnect()
        else:
            logf = open(self.run_case + '/log.Allrun', 'a+')
            Allrun = './' + self.run_case + '/' + 'Allrun.run'
            subprocess.Popen( Allrun, shell=True, stdout=logf, stderr=logf ).wait()

    def waiter(self):
        end_path_col = self.run_case + '/processors' + str(self.n_subdomains) + \
                '/' + (str(self.end_time) + '/').replace('.0/', '/')
        end_path_uncol = end_path_col.replace('processors' + str(self.n_subdomains), 'processor0')
        wait_time = 0
        while not os.path.exists(end_path_col) and not os.path.exists(end_path_uncol) \
              and wait_time < self.params['max_waiting_time']*60:
            time.sleep(1)
            wait_time += 1
        time.sleep(2)

    def read_results(self):
        start_time_add = (str(self.start_time) + '/').replace('.0/', '/')
        try:
            force = np.loadtxt(self.run_case + '/postProcessing/forces/' + start_time_add 
                               + 'coefficient.dat')
            time_episode = force[:, 0]
            if time_episode[-1] == self.end_time:
                self.converged = True
        except:
            pass

        if self.converged:
            episode_data = np.loadtxt(
                self.run_case + '/postProcessing/agentJet/' + start_time_add 
                + 'ActionState.dat', skiprows=2
            )
            self.action_times = episode_data[:,0]
            state_inds = time_episode.searchsorted(self.action_times)
            action_inds = np.append(state_inds[1:],-1)
            
            self.state_values = episode_data[:, 2:]
            self.action_values = episode_data[:, 1]

            for ind in action_inds:
                start_ind = max(ind - self.force_average_steps, 0)
                self.drag = np.append(self.drag, np.mean(force[start_ind:ind,1]))
                self.lift = np.append(self.lift, np.mean(force[start_ind:ind,3]))
            self.terminals = np.append(np.zeros(self.params['n_actions']-1, dtype=int), 2)          
        else:
            self.state_values = np.empty((0, self.params['n_states']), float)
            self.action_values = np.empty((0), float)
            self.terminals = np.empty((0), float)

    def compute_reward(self):
        if 'force' in self.params['reward_function']:
            penatly = self.params['reward_penalty']
            coeff = self.params['reward_coeff']
            self.reward_values = penatly - (self.drag + coeff*abs(self.lift))
        else:
            raise RuntimeError(
                'reward function {} not implemented'.format(self.params['reward_function'])
            )
  
    def set_foam_dict(self, dict, entry, value):
        command = ('foamDictionary ' + self.run_case + '/' 
            + dict + ' -entry ' + entry + ' -set '
            + str(value) + ' -disableFunctionEntries')
        subprocess.Popen(command, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, shell=True).wait()
